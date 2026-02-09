        # Courier解码使用的节点表征
        courier_nodes_embd = torch.cat((courier_depot_embd, courier_s1_pickup_embd, courier_s3_pickup_embd, courier_s1_delivery_embd, courier_s3_delivery_embd,), dim=1)

        # 构建可解码节点序列
        courier_decode_nodes_embd = torch.cat((
            courier_s1_pickup_embd,
            courier_s3_pickup_embd,
            courier_s1_delivery_embd,
            courier_s3_delivery_embd,
        ), dim=1)  # [batch_size, 4*n_request, n_embd]

        # 构建courier解码输入
        courier_task_target = couriers["task_target"]
        courier_target_embd = courier_nodes_embd[batch_arange[:, None], courier_task_target]
        invalid_task_mask = (courier_task_target < 0) | (courier_task_target >= courier_s3_delivery_end)
        courier_target_embd[invalid_task_mask] = self.NO_TARGET

        courier_decode_mask = couriers["time_left"] == 0
        courier_space = couriers["space"]
        # 这里可以搞成交替的
        comm_courier = torch.cat((courier_space, courier_target_embd.flatten(1)), -1)
        h_courier = torch.cat((
            courier_target_embd,
            courier_space[:, :, None],
            global_embd[:, None, :].expand(-1, n_courier, -1),
            comm_courier[:, None, :].expand(-1, n_courier, -1)
        ), -1)
        g_courier = self.courier_cross_attn.forward(
            self.courier_proj_h(h_courier),
            courier_decode_nodes_embd, # TODO 这里有一点出入，原代码没有去掉depot
            courier_decode_nodes_embd
        )
        q_courier = self.courier_to_q_g(g_courier)
        k_courier = self.courier_to_k_g(courier_decode_nodes_embd)
        # 采样和fleet handler
        D = 10
        d = self.n_embd / self.n_head
        k_courier_with_noaction = torch.cat((k_courier, self.NO_ACTION.expand(batch_size, 1, -1)), 1)
        u_courier = D * (q_courier @ k_courier_with_noaction.transpose(-1, -2) / d ** 0.5).tanh()
        
        # 更新courier_request状态
        cur_courier_request = _global["courier_request"].clone()
        cur_courier_space = courier_space.clone()
        assert ((cur_courier_request == 1).sum(-1) <= 1).all()
        # 处理已经到达pickup点的情况
        # cur_pickup_mask_courier = (couriers["time_left"] == 0) & (cur_courier_request == 1).any(-1)
        cur_pickup_mask_courier = (couriers["time_left"] == 0) & ((cur_courier_request == 1).sum(-1) == 1)
        if cur_pickup_mask_courier.any():
            pickup_batch_courier, pickup_courier = torch.where(cur_pickup_mask_courier)
            courier_pickup_request = (cur_courier_request[pickup_batch_courier, pickup_courier] == 1).nonzero(as_tuple=True)[1]  # TODO 这句不一样
            cur_courier_request[pickup_batch_courier, pickup_courier, courier_pickup_request] = 2
            cur_courier_space[pickup_batch_courier, pickup_courier] -= requests["volumn"][pickup_batch_courier, courier_pickup_request]

        # 构建courier mask（优先stage3）
        # TODO 这里+1有必要吗
        # TODO 检查这里阶段的检查对不对
        courier_mask = torch.ones(batch_size, n_courier, 4 * n_request + 1, device=device, dtype=torch.bool)
        # Stage1 pickup mask: request处于no_relation，courier在from点，有空间，vehicle可解码
        s1_pickup_mask = (
            (cur_courier_request == 0).all(1)[:, None, :] &  # no_relation
            (courier_decode_mask[:, :, None]) &  # vehicle可解码
            (cur_courier_space[:, :, None] >= requests["volumn"][:, None, :]) &  # 有空间
            stage1_mask[:, None, :]  # valid request
        )
        courier_mask[:, :, :n_request] = s1_pickup_mask
        # Stage3 pickup mask: request处于deliveryed_stage2（drone角度，即drone_request == 4），courier在station2点，有空间，vehicle可解码
        s3_pickup_mask = (
            (cur_courier_request == 0).all(1)[:, None, :] &  # courier角度no_relation
            (courier_decode_mask[:, :, None]) &  # vehicle可解码
            (cur_courier_space[:, :, None] >= requests["volumn"][:, None, :]) &  # 有空间
            stage3_mask[:, None, :]  # valid request
        )
        courier_mask[:, :, n_request:2*n_request] = s3_pickup_mask

        # Stage1 delivery mask: request处于to_delivery且courier的target == station1，courier携带该请求
        s1_delivery_mask = (
            (cur_courier_request == 2) &  # to_delivery
            (courier_decode_mask[:, :, None]) &  # vehicle可解码
            stage1_mask[:, None, :]  # valid request
        )
        courier_mask[:, :, 2*n_request:3*n_request] = s1_delivery_mask

        # Stage3 delivery mask: request处于to_delivery且courier的target == to，courier携带该请求
        s3_delivery_mask = (
            (cur_courier_request == 2)&  # to_delivery
            (courier_decode_mask[:, :, None]) &  # vehicle可解码
            stage3_mask[:, None, :]  # valid request
        )
        courier_mask[:, :, 3*n_request:4*n_request] = s3_delivery_mask

        # 优先stage3：如果stage3任务可用，将stage1对应的mask设为False（仅在同时有stage1和stage3时）
        # 对于每个courier，如果stage3 pickup可用，则mask掉对应的stage1 pickup
        # TODO 这是个提升性能的点，和原方法无关
        # for idx in range(n_courier):
        #     s3_pickup_available = courier_mask[:, idx, n_request:2*n_request]  # [batch_size, n_request]
        #     s1_pickup_available = courier_mask[:, idx, :n_request]  # [batch_size, n_request]
        #     # 如果同时有s1和s3可用，优先s3（mask掉s1）
        #     both_pickup_available = s1_pickup_available & s3_pickup_available
        #     courier_mask[:, idx, :n_request] = s1_pickup_available & ~both_pickup_available
            
        #     s3_delivery_available = courier_mask[:, idx, 3*n_request:4*n_request]
        #     s1_delivery_available = courier_mask[:, idx, 2*n_request:3*n_request]
        #     # 如果同时有s1和s3 delivery可用，优先s3（mask掉s1）
        #     both_delivery_available = s1_delivery_available & s3_delivery_available
        #     courier_mask[:, idx, 2*n_request:3*n_request] = s1_delivery_available & ~both_delivery_available

        
        u_courier[:, :, :-1].masked_fill_(~courier_mask[:, :, :-1], -torch.inf)
        u_courier[:, :, -1].masked_fill_((u_courier[:, :, :-1] == -torch.inf).all(-1), 1)
        p_courier = F.softmax(u_courier, -1)
        # 有事可做时必须执行任务
        p_courier = p_courier.masked_fill(torch.cat((
            torch.zeros_like(p_courier, dtype=torch.bool)[..., :-1],
            (p_courier[..., :-1].sum(-1) != 0).unsqueeze(-1)
        ), dim=-1), 0)

        cate_courier = Categorical(probs=p_courier)
        import pdb; pdb.set_trace()
        is_decoding = input_actions is None
        if is_decoding:
            sampled_new_task_target_courier = cate_courier.sample()

            new_task_target_courier = sampled_new_task_target_courier.clone()
            cur_courier_request_handler = cur_courier_request.clone()
            for idx in range(n_courier):
                # 处理stage1 pickup冲突检测
                s1_pickup_mask = courier_decode_mask[:, idx] & (new_task_target_courier[:, idx] < n_request)
                s1_pickup_batch = batch_arange[s1_pickup_mask]
                s1_pickup_request = new_task_target_courier[:, idx][s1_pickup_mask]
                no_conflict_s1 = (cur_courier_request_handler[s1_pickup_batch, :, s1_pickup_request] == 0).all(-1)
                new_task_target_courier[torch.where(s1_pickup_mask)[0][~no_conflict_s1], idx] = 4 * n_request  # NO_ACTION
                cur_courier_request_handler[s1_pickup_batch[no_conflict_s1], idx, s1_pickup_request[no_conflict_s1]] = 1
                
                # 处理stage3 pickup冲突检测
                s3_pickup_mask = courier_decode_mask[:, idx] & (new_task_target_courier[:, idx] >= n_request) & (new_task_target_courier[:, idx] < 2 * n_request)
                s3_pickup_batch = batch_arange[s3_pickup_mask]
                s3_pickup_request = new_task_target_courier[:, idx][s3_pickup_mask] - n_request  # 转换为request索引
                # no_conflict_s3 = (cur_courier_request_handler[s3_pickup_batch, :, s3_pickup_request] == 0).all(-1)
                # 冲突检测：只要没有其他courier处于to_pickup(1)或to_delivery(2)状态，就可以pickup
                # deliveryed_stage1(3)状态是允许存在的，因为那是上一阶段完成的状态
                conflict_mask = (cur_courier_request_handler[s3_pickup_batch, :, s3_pickup_request] == 1) | \
                                (cur_courier_request_handler[s3_pickup_batch, :, s3_pickup_request] == 2)
                no_conflict_s3 = ~conflict_mask.any(-1)
                new_task_target_courier[torch.where(s3_pickup_mask)[0][~no_conflict_s3], idx] = 4 * n_request  # NO_ACTION
                cur_courier_request_handler[s3_pickup_batch[no_conflict_s3], idx, s3_pickup_request[no_conflict_s3]] = 1

            # 解码courier actions
            courier_to = couriers["target"].clone()
            request_courier = torch.full([batch_size, n_request], n_courier, dtype=torch.int64, device=device)
            if cur_pickup_mask_courier.any():
                request_courier[pickup_batch_courier, courier_pickup_request] = pickup_courier
            
            # Stage1 pickup
            s1_pickup_mask_action = (new_task_target_courier < n_request) & courier_decode_mask
            courier_to[s1_pickup_mask_action] = requests["from"][
                batch_arange[:, None].masked_select(s1_pickup_mask_action),
                new_task_target_courier[s1_pickup_mask_action]
            ]
            
            # Stage3 pickup
            s3_pickup_mask_action = (new_task_target_courier >= n_request) & (new_task_target_courier < 2 * n_request) & courier_decode_mask
            courier_to[s3_pickup_mask_action] = requests["station2"][
                batch_arange[:, None].masked_select(s3_pickup_mask_action),
                new_task_target_courier[s3_pickup_mask_action] - n_request
            ]

            # Stage1 delivery
            s1_delivery_mask_action = (new_task_target_courier >= 2 * n_request) & (new_task_target_courier < 3 * n_request) & courier_decode_mask
            courier_to[s1_delivery_mask_action] = requests["station1"][
                batch_arange[:, None].masked_select(s1_delivery_mask_action),
                new_task_target_courier[s1_delivery_mask_action] - 2 * n_request
            ]

            # Stage3 delivery
            s3_delivery_mask_action = (new_task_target_courier >= 3 * n_request) & (new_task_target_courier < 4 * n_request) & courier_decode_mask
            courier_to[s3_delivery_mask_action] = requests["to"][
                batch_arange[:, None].masked_select(s3_delivery_mask_action),
                new_task_target_courier[s3_delivery_mask_action] - 3 * n_request
            ]

            # decode courier preassign
            courier_preassign = torch.full([batch_size, n_request], n_courier, dtype=torch.int64, device=device)
            courier_preassign[batch_arange[:, None].masked_select(s1_pickup_mask_action), new_task_target_courier[s1_pickup_mask_action]] = courier_arange.masked_select(s1_pickup_mask_action)
            courier_preassign[batch_arange[:, None].masked_select(s3_pickup_mask_action), new_task_target_courier[s3_pickup_mask_action] - n_request] = courier_arange.masked_select(s3_pickup_mask_action)
        else:
            new_task_target_courier = input_actions["new_task_target_courier"]
            courier_to = input_actions["courier"]
            request_courier = input_actions["request_courier"]
            sampled_new_task_target_courier = input_actions["sampled_new_task_target_courier"]
            courier_preassign = input_actions["courier_preassign"]
        log_prob_courier = cate_courier.log_prob(sampled_new_task_target_courier).sum(-1)
        entropy_courier = cate_courier.entropy().sum(-1)