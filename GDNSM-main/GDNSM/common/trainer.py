import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from time import time
from logging import getLogger
from GDNSM.utils.topk_evaluator import TopKEvaluator
from GDNSM.utils.utils import early_stopping, dict2str
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import copy  # <--- 新增
# 引入源码提供的工具
from GDNSM.torch_ema import ExponentialMovingAverage 
from torch.utils.tensorboard.writer import SummaryWriter
import os
import time

class Trainer:
    def __init__(self, config, model, diffusion_model=None, mg=False):
        """
        GDNSM Trainer，带 diffusion 生成负样本
        """
        self.config = config
        self.model = model
        self.diffusion_model = diffusion_model  # 生成负样本的模型
        self.mg = mg
        self.logger = getLogger()

        # 配置参数
        self.learner = config.get('learner', 'adam')
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epochs = config.get('epochs', 100)
        self.eval_step = min(config.get('eval_step', 1), self.epochs)
        self.stopping_step = config.get('stopping_step', 10)
        self.clip_grad_norm = config.get('clip_grad_norm', None)
        self.valid_metric = config.get('valid_metric', 'NDCG@20').lower()
        self.valid_metric_bigger = config.get('valid_metric_bigger', True)
        self.test_batch_size = config.get('eval_batch_size', 128)
        self.device = config.get('device', 'cuda')
        self.weight_decay = config.get('weight_decay', 0.0)
        if isinstance(self.weight_decay, str):
            self.weight_decay = eval(self.weight_decay)

        self.req_training = config.get('req_training', True)
        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {f'{j.lower()}@{k}': 0.0 for j, k in itertools.product(
            config.get('metrics', ['NDCG']), config.get('topk', [20])
        )}
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        # self.optimizer = self._build_optimizer()

        # lr_scheduler = getattr(config, 'learning_rate_scheduler', [0.96, 50])
        # fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        # self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)

        # === 新增两个优化器，分别更新 φ 和 θ ===
        theta_params = list(self.model.diffusion_MM.parameters())
        theta_param_ids = set(id(p) for p in theta_params)
        phi_params = [p for p in self.model.parameters() if id(p) not in theta_param_ids]

        self.opt_phi = optim.Adam(phi_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        self.opt_theta = optim.Adam(theta_params, lr=self.learning_rate, weight_decay=self.weight_decay)

        # === 各自配一个 scheduler ===
        lr_scheduler = config.get('learning_rate_scheduler', [0.96, 50])
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])

        self.lr_phi = optim.lr_scheduler.LambdaLR(self.opt_phi, lr_lambda=fac)
        self.lr_theta = optim.lr_scheduler.LambdaLR(self.opt_theta, lr_lambda=fac)


        self.eval_type = config.get('eval_type', 'full')
        self.evaluator = TopKEvaluator(config)

        # GDNSM 多目标训练参数
        self.alpha1 = config.get('alpha1', 1.0)
        self.alpha2 = config.get('alpha2', 1.0)
        self.beta = config.get('beta', 1)
        self.gamma = config.get('gamma',1)

        # ==========================================
        # # [新增] UFNRec 模块初始化
        # # ==========================================
        # print("Initializing UFNRec Module...")

        # # 1. 克隆教师模型 (Teacher Model)
        # # 为什么要改：UFNRec 需要一个稳定的 Teacher 来监督 Student，防止 Student 在翻转标签后学偏。
        # # Teacher 不参与梯度更新，所以 requires_grad=False。
        # self.teacher_model = copy.deepcopy(self.model)
        # for p in self.teacher_model.parameters():
        #     p.requires_grad = False
        # self.teacher_model.eval() # 永远是 eval 模式

        # # 2. UFNRec 超参数
        # # m: 连续多少次高分才认定为 False Negative (挖掘阈值)
        # self.ufn_m = getattr(config, 'ufn_m', 2) 
        # # alpha: Consistency Loss 的权重
        # self.ufn_alpha = getattr(config, 'ufn_alpha', 0.1)
        # # decay: EMA 更新的衰减率 (Teacher 走得多慢)
        # self.ufn_decay = getattr(config, 'ufn_ema_decay', 0.999)

        # # 3. 嫌疑人计数器 (Suspect Buffer)
        # # 为什么要改：我们需要记录每个 (User, Item) 出现了多少次“分数倒挂”。
        # # 使用字典 {(user_id, item_id): count} 来节省内存。
        # self.fn_counter = {} 

        self.logger = getLogger()

        # ==========================================
        # [新增] TensorBoard 初始化
        # ==========================================
        # 根据当前时间生成唯一的日志目录，防止覆盖
        timestamp = time.strftime('%m%d_%H%M')
        # 目录结构: runs/数据集_模型_时间
        log_dir = os.path.join("runs", f"{config['dataset']}_GDNSM_UFN_{timestamp}")
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard log directory: {log_dir}")
        # ==========================================
        #__getitem__
        # ================= [UFNRec Init] =================
        self.ufn_warmup = config.get('ufn_warmup', 5) # 热身 Epoch
        self.ufn_m = config.get('reverse', 2)        # 阈值 m
        self.ufn_alpha = config.get('lbd', 0.3)       # Consistency Loss 权重
        self.ufn_decay = config.get('decay', 0.999)   # EMA decay

        # [GDNSM 扩散模型参数 - 修复此处]后面从self.config中导入
        # self.sched_S = config.get('sched_S', 30) 
        # self.sched_lambda = config.get('sched_lambda', 1)
        # self.num_M = config.get('num_M', 5)
        
        # [新增] 读取 d_epoch 和 use_mm_diff
        self.d_epoch = config.get('d_epoch', 1)       # 默认训练 5 次
        self.use_mm_diff = config.get('use_mm_diff', True) # 默认开启

        # [优化1] 使用 EMA 对象，而不是 Teacher Model
        # 注意：这里只追踪 phi (MCI Encoder) 的参数，不追踪 diffusion
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ufn_decay)
        self.ema.to(self.device)
        
        # [优化2] 双层缓存机制
        # hard_items: {(uid, iid): True}  记录上一轮的嫌疑人
        # cnt_items: {(uid, iid): count}  记录累计次数
        self.hard_items = {} 
        self.cnt_items = {}

        # ==========================================


    # # ==========================================
    # # [新增] EMA 更新函数
    # # ==========================================
    # def _update_teacher(self):
    #     """
    #     用 Student (self.model) 的参数更新 Teacher (self.teacher_model)
    #     公式: theta_teacher = decay * theta_teacher + (1 - decay) * theta_student
    #     """
    #     with torch.no_grad():
    #         for param_t, param_s in zip(self.teacher_model.parameters(), self.model.parameters()):
    #             param_t.data.mul_(self.ufn_decay)
    #             param_t.data.add_((1 - self.ufn_decay) * param_s.data)
    # # ==========================================

    def _build_optimizer(self):
        opt_dict = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'adagrad': optim.Adagrad,
            'rmsprop': optim.RMSprop
        }
        OptimCls = opt_dict.get(self.learner.lower(), optim.Adam)
        if OptimCls == optim.Adam and self.learner.lower() not in opt_dict:
            self.logger.warning('Unrecognized optimizer, using default Adam')
        return OptimCls(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False):
        self.model.eval()
        batch_matrix_list = []
        for batch in eval_data:
            scores = self.model.full_sort_predict(batch)
            masked_items = batch[1]
            scores[masked_items[0], masked_items[1]] = -1e10
            _, topk_index = torch.topk(scores, max(self.config.get('topk', [20])), dim=-1)
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test)

    # def _train_epoch_bprcl_diffu(self, train_data):
    #     self.model.train()
    #     total_loss = 0.0

    #     # 配置参数
    #     d_epoch = getattr(self.config, 'd_epoch', 1)
    #     use_mm_diff = getattr(self.config, 'use_mm_diff', True)
    #     base_cl_weight = getattr(self.config, 'cl_weight', 0.01)
    #     max_cl_weight = getattr(self.config, 'max_cl_weight', 0.1)
    #     beta = getattr(self.config, 'beta', 1.0)
    #     epoch = getattr(self, 'current_epoch', 0)
    #     E = max(getattr(self.config, 'epochs', 100), 1)
    #     cl_weight = base_cl_weight + (max_cl_weight - base_cl_weight) * (epoch / E)

    #     for batch in train_data:
    #         users = batch[0]
    #         pos_items = batch[1]

    #         if len(batch) >= 3:
    #             neg_items = batch[2]
    #         else:
    #             num_items = self.model.n_items
    #             neg_items = torch.randint(0, num_items, pos_items.shape, device=pos_items.device)

    #         # ===== Phase A: Diffusion θ =====
    #         if use_mm_diff:
    #             for p in self.model.parameters():
    #                 p.requires_grad = False
    #             for p in self.model.diffusion_MM.parameters():
    #                 p.requires_grad = True

    #             for _ in range(d_epoch):
    #                 with torch.no_grad():
    #                     ua_embeddings, ia_embeddings, _, _ = self.model.forward(self.model.norm_adj, train=True)
    #                     u_g = ua_embeddings[users]
    #                     x0 = ia_embeddings[pos_items]

    #                     t_cond = self.model.text_trs(self.model.text_feat[pos_items])
    #                     v_cond = self.model.image_trs(self.model.image_feat[pos_items])
    #                     labels = torch.cat([u_g, t_cond, v_cond], dim=1)

    #                 diff_loss = self.model.diffusion_MM(x0, labels, device=x0.device)

    #                 self.opt_theta.zero_grad()
    #                 diff_loss.backward()
    #                 self.opt_theta.step()

    #             for p in self.model.parameters():
    #                 p.requires_grad = True
    #             for p in self.model.diffusion_MM.parameters():
    #                 p.requires_grad = False

    #         # ===== Phase B: φ (BPR + CL + Diffusion Negative) =====
    #         ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.model.forward(self.model.norm_adj, train=True)
    #         u_g = ua_embeddings[users]
    #         pos_g = ia_embeddings[pos_items]
    #         rand_neg_g = ia_embeddings[neg_items]

    #         # -------- Contrastive Learning Loss --------
    #         side_u, side_i = torch.split(side_embeds, [self.model.n_users, self.model.n_items], dim=0)
    #         cont_u, cont_i = torch.split(content_embeds, [self.model.n_users, self.model.n_items], dim=0)
    #         side_u, side_i = F.normalize(side_u, dim=1), F.normalize(side_i, dim=1)
    #         cont_u, cont_i = F.normalize(cont_u, dim=1), F.normalize(cont_i, dim=1)
    #         cl_loss = self.model.InfoNCE(side_i[pos_items], cont_i[pos_items], 0.2) + \
    #                 self.model.InfoNCE(side_u[users], cont_u[users], 0.2)

    #         # -------- Diffusion生成负样本并应用动态难度调度 --------
    #         L_NEG = torch.tensor(0.0, device=pos_g.device)
    #         if use_mm_diff:
    #             neg_sets = []
    #             with torch.no_grad():
    #                 t_cond = self.model.text_trs(self.model.text_feat[pos_items])
    #                 v_cond = self.model.image_trs(self.model.image_feat[pos_items])
    #                 labels_gen = torch.cat([u_g, t_cond, v_cond], dim=1)

    #                 # 生成三种难度的负样本 (Difficulty level 1-1, 1-2, 2)
    #                 for flag in (1, 2, 3):
    #                     steps = self.model.diffusion_MM.sample(pos_g.shape, labels_gen, flag=flag)
    #                     neg_sets.append(steps[-1])  # 取最后一步作为负样本

    #             all_negs = torch.stack(neg_sets, dim=1)  # shape: [batch, difficulty, dim]
    #             u_norm = F.normalize(u_g, dim=1).unsqueeze(1)
    #             n_norm = F.normalize(all_negs, dim=2)
    #             cos = (u_norm * n_norm).sum(dim=2)
    #             _, order = torch.sort(cos, dim=1, descending=False)  # 从最相似(困难)到最不相似(简单)

    #             # 动态难度调度 (Curriculum Learning)
    #             Mmax = 3
    #             g = max(1, min(Mmax, int(round(Mmax * (epoch+1) / E))))  # 随 epoch 增加选择更多难样本
    #             for k in range(g):
    #                 idx = order[:, k]
    #                 pick = all_negs[torch.arange(all_negs.size(0)), idx]
    #                 mf, _, _ = self.model.bpr_loss(u_g, pos_g, pick)
    #                 L_NEG += mf
    #             L_NEG /= g

    #         # -------- 原始 BPR loss --------
    #         bpr_mf, bpr_emb, bpr_reg = self.model.bpr_loss(u_g, pos_g, rand_neg_g)

    #         # -------- 总 loss --------
    #         total = bpr_mf + bpr_emb + bpr_reg + cl_weight * cl_loss
    #         if use_mm_diff:
    #             total += beta * L_NEG

    #         self.opt_phi.zero_grad()
    #         total.backward()
    #         self.opt_phi.step()

    #         total_loss += total.item()

    #     self.current_epoch = epoch + 1
    #     return total_loss / len(train_data)
    def _train_epoch_phi(self, train_data, epoch):
        self.model.train()
        total_loss = 0.0
        for batch in train_data:
            users, pos_items = batch[0], batch[1]
            neg_items = torch.randint(0, self.model.n_items, pos_items.shape, device=pos_items.device)

            ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.model.forward(self.model.norm_adj, train=True)
            u_g, pos_g, neg_g = ua_embeddings[users], ia_embeddings[pos_items], ia_embeddings[neg_items]

            # BPR
            bpr_mf, bpr_emb, bpr_reg = self.model.bpr_loss(u_g, pos_g, neg_g)

            # CL
            side_u, side_i = torch.split(side_embeds, [self.model.n_users, self.model.n_items], dim=0)
            cont_u, cont_i = torch.split(content_embeds, [self.model.n_users, self.model.n_items], dim=0)
            cl_loss = self.model.InfoNCE(side_i[pos_items], cont_i[pos_items], 0.2) + \
                    self.model.InfoNCE(side_u[users], cont_u[users], 0.2)

            total = bpr_mf + bpr_emb + bpr_reg +  0.01 *cl_loss

            self.opt_phi.zero_grad()
            total.backward()
            self.opt_phi.step()
            total_loss += total.item()

        return total_loss / len(train_data)

    def _train_epoch_theta(self, train_data, epoch):
        # 1. 如果配置关闭了扩散模型，直接返回 0 Loss
        if not self.use_mm_diff:
            return 0.0
        
        self.model.train()
        total_loss = 0.0

        # φ (Encoder) 冻结，只训练 θ (Diffusion)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.diffusion_MM.parameters():
            p.requires_grad = True

        for batch in train_data:
            users, pos_items = batch[0], batch[1]

            # 预先计算条件 Embedding (因为 Encoder 被冻结了，这部分对于 d_epoch 循环是固定的)
            # 放在循环外可以稍微节省一点点计算量
            with torch.no_grad():
                ua_embeddings, ia_embeddings, _, _ = self.model.forward(self.model.norm_adj, train=True)
                u_g = ua_embeddings[users]
                x0 = ia_embeddings[pos_items]

                # 获取多模态特征作为条件
                t_cond = self.model.text_trs(self.model.text_feat[pos_items])
                v_cond = self.model.image_trs(self.model.image_feat[pos_items])
                labels = torch.cat([u_g, t_cond, v_cond], dim=1)

            # [关键修复] 启用 d_epoch 循环
            # 对当前 Batch 反复训练 d_epoch 次
            for _ in range(self.d_epoch):
                # 注意：虽然 x0 和 labels 是一样的，但 diffusion_MM 内部会随机采样不同的 timestep t 和噪声
                # 所以这多次训练是有意义的
                diff_loss = self.model.diffusion_MM(x0, labels, device=x0.device)
                
                self.opt_theta.zero_grad()
                diff_loss.backward()
                self.opt_theta.step()
                
                total_loss += diff_loss.item()

        # 解冻回来
        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.model.diffusion_MM.parameters():
            p.requires_grad = False

        # 计算平均 Loss (总 Loss / (Batch数 * d_epoch))
        return total_loss / (len(train_data) * self.d_epoch)

    def _train_epoch_joint(self, train_data, epoch):
        self.model.train()
        # total_loss = 0.0

        # [修改] 定义分项 Loss 累加器
        epoch_loss_total = 0.0
        epoch_loss_bpr = 0.0
        epoch_loss_cl = 0.0
        epoch_loss_lneg = 0.0
        epoch_loss_ufn = 0.0
        epoch_loss_ufn_bce = 0.0 # 监控平反力度
        epoch_loss_ufn_con = 0.0 # 监控教师约束

        # 用于保存相似度
        cos_sim_pos_list = []
        cos_sim_vneg_list = []
        cos_sim_tneg_list = []
        cos_sim_tvneg_list = []

        # 动态难度调度参数
        S = self.config.get('smoothing_S', 10)        # pacing 参数 S
        lam = self.config.get('lambda_ds', 1)  # pacing 参数 λ
        M = self.config.get('num_generated_neg', 3)           # 基础负样本数 (总共 3M)

        def g(ep):
            if ep < S:
                return 0
            else:
                return int(min(3 * M, ((ep / S) ** lam) * M))

        
        # 判断是否开启 UFNRec (热身结束)
        ufn_flag = epoch >= self.ufn_warmup

        for batch in train_data:
            users, pos_items = batch[0], batch[1]
            
            # [关键] 必须在这里生成随机负样本 ID，因为 UFNRec 需要 ID 来查表
            neg_items = torch.randint(0, self.model.n_items, (pos_items.shape[0],), device=pos_items.device)
            
            # ================= [Step 1: 判定反转名单] =================
            # 在 Forward 之前，先检查哪些随机负样本是"老熟人"(FN)
            reverse_indices = [] # 记录 batch 内的下标 [0, 5, 10...]
            
            if ufn_flag:
                # 遍历当前 batch 的随机负样本
                # 注意：这里需要在 CPU 上做字典查询，稍微有点慢，但必须这么做
                cpu_users = users.cpu().numpy()
                cpu_negs = neg_items.cpu().numpy()
                
                for idx, (uid, iid) in enumerate(zip(cpu_users, cpu_negs)):
                    key = (uid, iid)
                    # 只有当它在上一轮被标记为 Hard，且累计次数够多时
                    if key in self.hard_items:
                        # 增加计数 (源码逻辑：每次遇到都加)
                        self.cnt_items[key] = self.cnt_items.get(key, 0) + 1
                        
                        if self.cnt_items[key] >= self.ufn_m:
                            reverse_indices.append(idx)
                
                # [添加这行 Debug]
                if len(reverse_indices) > 0:
                    print(f"Epoch {epoch}: 终于挖到了 {len(reverse_indices)} 个宝藏！")
                else:
                    # 你大概率会看到满屏的这个
                    # print("Epoch {epoch}: 空空如也...") 
                    pass
            # ================= [Step 2: Forward] =================            
            
            
            # 1. 正向传播 (Student)
            # 注意：我们需要拿到 neg_g (真实随机负样本向量)
            ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.model.forward(self.model.norm_adj, train=True)
            u_g = ua_embeddings[users]
            pos_g = ia_embeddings[pos_items]
            neg_g = ia_embeddings[neg_items] # <--- 这是我们要挖掘的对象
            
            # ================= [Step 3: 计算 UFN Loss] =================
            # ==========================================
            # [插入] UFNRec: 挖掘与平反逻辑
            # ==========================================
            loss_bce = torch.tensor(0.0, device=self.device)
            loss_con = torch.tensor(0.0, device=self.device)
            loss_ufn = torch.tensor(0.0, device=self.device)

            if ufn_flag and len(reverse_indices) > 0:
                rev_idx_tensor = torch.tensor(reverse_indices, device=self.device)
                
                # 1. 拿到 FN 样本
                u_fn = u_g[rev_idx_tensor]
                i_fn = neg_g[rev_idx_tensor]
                
                # 2. BCE Loss (Label Reversing)
                logits = (u_fn * i_fn).sum(dim=1)
                loss_bce = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))

                # 3. Consistency Loss (with EMA)
                # [优化1] 使用 context manager 切换权重
                with self.ema.average_parameters():
                    # 此时 self.model 的权重变成了 Teacher 的权重
                    # 我们只需要重新计算这几个样本的 embedding
                    # 注意：为了省显存，这里最好只算必要的，但 LightGCN 特性决定了必须全量 forward
                    # 如果显存不够，这里可以 wrap 在 torch.no_grad() 里
                    t_ua, t_ia, _, _ = self.model.forward(self.model.norm_adj, train=True)
                    t_u_fn = t_ua[users[rev_idx_tensor]]
                    t_i_fn = t_ia[neg_items[rev_idx_tensor]]
                    
                    teacher_logits = (t_u_fn * t_i_fn).sum(dim=1)
                    teacher_probs = torch.sigmoid(teacher_logits)
                
                # 切换回来后，self.model 变回 Student
                loss_con = F.binary_cross_entropy_with_logits(logits, teacher_probs)
                
                loss_ufn = loss_bce + self.ufn_alpha * loss_con


            # # A. 挖掘 (Mining)
            # with torch.no_grad():
            #     # 计算分数
            #     pos_scores = (u_g * pos_g).sum(dim=1)
            #     neg_scores = (u_g * neg_g).sum(dim=1)
                
            #     # 找到“倒挂”的样本 (负样本得分 > 正样本得分)
            #     # 这里的逻辑是：如果模型觉得这个随机负样本比正样本还好，它很可能是 False Negative
            #     suspect_mask = neg_scores > pos_scores
            #     suspect_indices = torch.nonzero(suspect_mask).squeeze() # tensor([1, 3])  # LongTensor，表示第1个和第3个样本是嫌疑人

            #     # 更新计数器
            #     confirmed_fn_indices = [] # 本次 batch 确认的 FN 索引
            #     if suspect_indices.numel() > 0: # numel() 是 number of elements (元素个数)
            #         # 处理 scalar 这里的坑
            #         if suspect_indices.dim() == 0: suspect_indices = suspect_indices.unsqueeze(0)
                    
            #         for idx in suspect_indices:
            #             idx = idx.item()
            #             uid = users[idx].item()
            #             iid = neg_items[idx].item()
            #             key = (uid, iid)
                        
            #             # 计数 +1
            #             self.fn_counter[key] = self.fn_counter.get(key, 0) + 1
                        
            #             # 判断是否达到阈值 m
            #             if self.fn_counter[key] >= self.ufn_m:
            #                 confirmed_fn_indices.append(idx)
            # B. 计算平反 Loss (Reversing & Consistency)
            # if len(confirmed_fn_indices) > 0:
            #     # 拿到被确认为 FN 的样本向量
            #     fn_idx_tensor = torch.tensor(confirmed_fn_indices, device=self.device)
            #     u_fn = u_g[fn_idx_tensor]
            #     i_fn = neg_g[fn_idx_tensor] # 这些原本是负样本，现在我们要给它平反

            #     # --- Loss 1: BCE Loss (Label Reversing) ---
            #     # 强行把 Label 设为 1
            #     logits = (u_fn * i_fn).sum(dim=1)
            #     labels = torch.ones_like(logits)
            #     loss_ufn_bce = F.binary_cross_entropy_with_logits(logits, labels)
                
            #     # --- Loss 2: Consistency Loss (Teacher Supervision) ---
            #     # 让 Teacher 也来看看这些样本
            #     with torch.no_grad():
            #         # Teacher 也要做一次 forward 拿到最新的 Embedding
            #         t_ua, t_ia, _, _ = self.teacher_model.forward(self.model.norm_adj, train=True)
            #         t_u_fn = t_ua[users[fn_idx_tensor]]
            #         t_i_fn = t_ia[neg_items[fn_idx_tensor]]
                    
            #         # Teacher 的打分 (Soft Label)
            #         teacher_logits = (t_u_fn * t_i_fn).sum(dim=1)
            #         teacher_probs = torch.sigmoid(teacher_logits)

            #     # Student 的预测概率
            #     # student_probs = torch.sigmoid(logits)
                
            #     # 计算一致性 Loss (这里用 BCE 形式的蒸馏)
            #     # 目标是让 Student 的概率去逼近 Teacher 的概率
            #     # loss_ufn_con = -(teacher_probs * torch.log(student_probs + 1e-8) + 
            #     #                  (1 - teacher_probs) * torch.log(1 - student_probs + 1e-8)).mean()
                
            #     # 注意：这里传入的是 student_logits (未激活)，而不是 student_probs
            #     loss_ufn_con = F.binary_cross_entropy_with_logits(logits, teacher_probs)
            # # ==========================================


            # 2. 原有的 Loss 计算 (保持不变)
            bpr_mf, bpr_emb, bpr_reg = self.model.bpr_loss(u_g, pos_g, neg_g)

            # CL
            side_u, side_i = torch.split(side_embeds, [self.model.n_users, self.model.n_items], dim=0)
            cont_u, cont_i = torch.split(content_embeds, [self.model.n_users, self.model.n_items], dim=0)
            cl_loss = self.model.InfoNCE(side_i[pos_items], cont_i[pos_items], 0.2) + \
                    self.model.InfoNCE(side_u[users], cont_u[users], 0.2)

            # θ 生成负样本 (固定 θ)
            with torch.no_grad():
                t_cond = self.model.text_trs(self.model.text_feat[pos_items])
                v_cond = self.model.image_trs(self.model.image_feat[pos_items])
                labels_gen = torch.cat([u_g, t_cond, v_cond], dim=1)

                # 三种生成负样本
                O_v_all, O_t_all, O_tv_all = [], [], []
                for _ in range(M):
                    O_v_all.append(self.model.diffusion_MM.sample(pos_g.shape, labels_gen, flag=1)[-1])
                    O_t_all.append(self.model.diffusion_MM.sample(pos_g.shape, labels_gen, flag=2)[-1])
                    O_tv_all.append(self.model.diffusion_MM.sample(pos_g.shape, labels_gen, flag=3)[-1])


            with torch.no_grad(): 
                # 计算余弦相似度
                u_norm = F.normalize(u_g, dim=1)             # [B, D]
                pos_norm = F.normalize(pos_g, dim=1)
                for vneg, tneg, tvneg in zip(O_v_all, O_t_all, O_tv_all):
                    vneg_norm = F.normalize(vneg, dim=1)
                    tneg_norm = F.normalize(tneg, dim=1)
                    tvneg_norm = F.normalize(tvneg, dim=1)
                    cos_sim_pos_list.extend((u_norm * pos_norm).sum(dim=1).cpu().numpy())
                    cos_sim_vneg_list.extend((u_norm * vneg_norm).sum(dim=1).cpu().numpy())
                    cos_sim_tneg_list.extend((u_norm * tneg_norm).sum(dim=1).cpu().numpy())
                    cos_sim_tvneg_list.extend((u_norm * tvneg_norm).sum(dim=1).cpu().numpy())

            # 动态难度调度
            Q_diff = O_v_all + O_t_all + O_tv_all
            Q_diff = Q_diff[:g(epoch+1)]


            # 计算 L_NEG
            L_NEG = 0.0
            for hard_neg in Q_diff:
                mf_neg, _, _ = self.model.bpr_loss(u_g, pos_g, hard_neg)
                L_NEG += mf_neg
            if len(Q_diff) > 0:
                L_NEG /= len(Q_diff)
            if epoch<S:
                L_NEG = 0.0

            # 总损失
            # [修改] 加入 UFNRec 的 loss
            total = bpr_mf + bpr_emb + bpr_reg + 0.01* cl_loss + self.beta * L_NEG + self.gamma * loss_ufn
                    
            self.opt_phi.zero_grad()
            total.backward()
            self.opt_phi.step()

            # [优化1] 更新 EMA
            if ufn_flag:
                if self.ema.shadow_params[0].device != next(self.model.parameters()).device:
                    self.ema.to(next(self.model.parameters()).device)
                    
                self.ema.update()


            # ================= [Step 5: Mining (事后挖掘)] =================
            # [优化2] 在参数更新后，利用最新的 Logits 更新 hard_items
            if ufn_flag:
                with torch.no_grad():
                    # 重新算分 (用更新后的参数，或者直接复用 backward 前的 logits 也可以，源码是用 backward 前的)
                    # 为了效率，我们复用 backward 前的 u_g, pos_g, neg_g
                    # 虽然参数更新了一点点，但影响不大
                    pos_scores = (u_g * pos_g).sum(dim=1)
                    neg_scores = (u_g * neg_g).sum(dim=1)
                    
                    # 找到倒挂
                    suspect_mask = neg_scores > pos_scores
                    suspect_indices = torch.nonzero(suspect_mask).squeeze()
                    
                    if suspect_indices.numel() > 0:
                        if suspect_indices.dim() == 0: suspect_indices = suspect_indices.unsqueeze(0)
                        
                        # 清空上一轮的 hard_items? 源码逻辑是动态维护
                        # 这里我们简单点：每轮都把新发现的加进去
                        cpu_idx = suspect_indices.cpu().numpy()
                        for idx in cpu_idx:
                            uid = users[idx].item()
                            iid = neg_items[idx].item()
                            # 记录到 hard_items，这就成为了下一轮的候选人
                            self.hard_items[(uid, iid)] = True
            
            # [监控] 累加本 Batch 的 Loss
            epoch_loss_total += total.item()
            epoch_loss_bpr += bpr_mf.item()
            epoch_loss_cl += cl_loss.item()
            epoch_loss_lneg += L_NEG.item() if isinstance(L_NEG, torch.Tensor) else L_NEG
            epoch_loss_ufn += loss_ufn.item()
            epoch_loss_ufn_bce += loss_bce.item()
            epoch_loss_ufn_con += loss_con.item()


            # total_loss += total.item()

        # 打印余弦相似度分布
        print("Cosine similarity distributions:")
        print(f"Positive samples: mean={np.mean(cos_sim_pos_list):.4f}, std={np.std(cos_sim_pos_list):.4f}")
        print(f"Video Negatives: mean={np.mean(cos_sim_vneg_list):.4f}, std={np.std(cos_sim_vneg_list):.4f}")
        print(f"Text Negatives: mean={np.mean(cos_sim_tneg_list):.4f}, std={np.std(cos_sim_tneg_list):.4f}")
        print(f"Video+Text Negatives: mean={np.mean(cos_sim_tvneg_list):.4f}, std={np.std(cos_sim_tvneg_list):.4f}")
        
        # ==========================================
        # [核心] 写入 TensorBoard
        # ==========================================
        num_batches = len(train_data)
        self.writer.add_scalar('Loss/Total', epoch_loss_total / num_batches, epoch)
        self.writer.add_scalar('Loss/BPR_Basic', epoch_loss_bpr / num_batches, epoch)
        self.writer.add_scalar('Loss/CL', epoch_loss_cl / num_batches, epoch)
        self.writer.add_scalar('Loss/Diffusion_Gen', epoch_loss_lneg / num_batches, epoch)
        
        # 重点关注这三个！如果不为0，说明生效了
        self.writer.add_scalar('Loss/UFN_Total', epoch_loss_ufn / num_batches, epoch)
        self.writer.add_scalar('Loss/UFN_BCE', epoch_loss_ufn_bce / num_batches, epoch)
        self.writer.add_scalar('Loss/UFN_Consistency', epoch_loss_ufn_con / num_batches, epoch)
        # ==========================================

        return epoch_loss_total / len(train_data)



    # def fit(self, train_data, valid_data=None, test_data=None, saved=True, verbose=True):
    #     """
    #     三阶段训练:
    #     1. 推荐模型 φ 预训练 (BPR+CL)
    #     2. 扩散模型 θ 训练 (φ 固定)
    #     3. 联合阶段 (θ 固定, φ 更新, 生成负样本)
    #     """
    #     E1 = getattr(self.config, 'pretrain_epochs', 10)
    #     E2 = getattr(self.config, 'diff_epochs', 10)
    #     E3 = getattr(self.config, 'joint_epochs', 30)

    #     self.best_valid_score = -1e9 if self.valid_metric_bigger else 1e9
    #     self.best_valid_result, self.best_test_upon_valid = {}, {}

    #     print(f"\n===> 开始三阶段训练，总轮数 {E1+E2+E3}")

    #     # -------- 阶段 1: 推荐模型 φ --------
    #     print(f"\n[阶段 1] 预训练推荐模型 φ ({E1} 轮)")
    #     for epoch in range(E1):
    #         loss = self._train_epoch_phi(train_data, epoch)
    #         if verbose:
    #             self.logger.info(f"[阶段 1] Epoch {epoch+1}/{E1} | Loss={loss:.4f}")
    #         # 验证
    #         if valid_data and (epoch + 1) % self.eval_step == 0:
    #             self._do_validation(epoch, valid_data, test_data, saved, verbose)

    #     # -------- 阶段 2: 扩散模型 θ --------
    #     print(f"\n[阶段 2] 训练扩散模型 θ (直到收敛, φ 固定)")

    #     patience = getattr(self.config, 'diff_patience', 5)   # 连续多少次没提升就停
    #     min_delta = getattr(self.config, 'diff_min_delta', 1e-2)  # 最小改善幅度
    #     best_loss = float('inf')
    #     bad_count = 0
    #     epoch = 0

    #     while True:
    #         loss = self._train_epoch_theta(train_data, epoch)
    #         if verbose:
    #             self.logger.info(f"[阶段 2] Epoch {epoch+1} | Diff Loss={loss:.4f}")

    #         # 验证
    #         if valid_data and (epoch + 1) % self.eval_step == 0:
    #             self._do_validation(epoch, valid_data, test_data, saved, verbose)

    #         # 收敛判断
    #         if best_loss - loss > min_delta:
    #             best_loss = loss
    #             bad_count = 0
    #         else:
    #             bad_count += 1

    #         if bad_count >= patience:
    #             print(f"θ 训练在 {epoch+1} 轮时提前收敛 ✅ (最优 Loss={best_loss:.4f})")
    #             break

    #         epoch += 1


    #     # -------- 阶段 3: 联合阶段 --------
    #     print(f"\n[阶段 3] 联合训练 φ+θ ({E3} 轮, θ 固定)")
    #     for epoch in range(E3):
    #         loss = self._train_epoch_joint(train_data, epoch)
    #         if verbose:
    #             self.logger.info(f"[阶段 3] Epoch {epoch+1}/{E3} | Joint Loss={loss:.4f}")
    #         if valid_data and (epoch + 1) % self.eval_step == 0:
    #             self._do_validation(epoch, valid_data, test_data, saved, verbose)

    #     print("\n===> 三阶段训练完成 ✅")
    #     return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid

    def _do_validation(self, epoch, valid_data, test_data, saved=True, verbose=True):
        valid_result = self.evaluate(valid_data)
        valid_score = valid_result.get(self.valid_metric, valid_result.get('NDCG@20', 0.0))
        test_result = self.evaluate(test_data, is_test=True) if test_data else None

        # ==========================================
        # [新增] 写入验证集指标
        # ==========================================
        # valid_result 通常是一个字典，包含 'Recall@10', 'NDCG@20' 等
        for metric, value in valid_result.items():
            self.writer.add_scalar(f'Metric/Valid/{metric}', value, epoch)
        
        # 如果有测试集结果，也写进去
        if test_data and test_result:
            for metric, value in test_result.items():
                self.writer.add_scalar(f'Metric/Test/{metric}', value, epoch)
        # ==========================================
        

        # 判断是否更新 best
        is_better = (valid_score > self.best_valid_score) if self.valid_metric_bigger else (valid_score < self.best_valid_score)
        if is_better:
            self.best_valid_score = valid_score
            self.best_valid_result = valid_result
            self.best_test_upon_valid = test_result
            if saved:
                save_path = f'best_model_{self.config.get("model", "GDNSM")}.pth'
                torch.save(self.model.state_dict(), save_path)
                self.logger.info(f'Best model saved to {save_path}')

        if verbose:
            self.logger.info(f"Epoch {epoch} | Valid: {valid_score:.4f}")
            if test_result:
                self.logger.info(f"Test Result: {dict2str(test_result)}")
        # [修改] 返回是否变得更好 (True/False)
        return is_better
    
    def fit(self, train_data, valid_data=None, test_data=None, saved=True, verbose=True):
        """
        每个 epoch 都执行三个阶段:
        1. 更新推荐模型 φ (BPR+CL)
        2. 更新扩散模型 θ (φ 冻结)
        3. 联合训练 (θ 固定, φ 更新, 生成负样本)
        """
        E = self.config.get('total_epochs', 100)  # 总 epoch 数
        self.best_valid_score = -1e9 if self.valid_metric_bigger else 1e9
        self.best_valid_result, self.best_test_upon_valid = {}, {}

        # [早停参数]
        patience = self.stopping_step  # 从 config 读取，默认通常是 10 或 20
        wait = 0                       # 忍耐计数器

        print(f"\n===> 开始联合训练 (每个 epoch 包含 3 个阶段)，总轮数 {E}")

        for epoch in range(E):
            # -------- 阶段 1: 推荐模型 φ --------
            loss_phi = self._train_epoch_phi(train_data, epoch)
            
            # -------- 阶段 2: 扩散模型 θ --------
            loss_theta = self._train_epoch_theta(train_data, epoch)

            # -------- 阶段 3: 联合训练 --------
            loss_joint = self._train_epoch_joint(train_data, epoch)

            if verbose:
                self.logger.info(
                    f"Epoch {epoch+1}/{E} | "
                    f"Phi Loss={loss_phi:.4f} | "
                    f"Theta Loss={loss_theta:.4f} | "
                    f"Joint Loss={loss_joint:.4f}"
                )

            # 验证
            if valid_data and (epoch + 1) % self.eval_step == 0:
                # 接收返回值：是否变好了？
                is_better = self._do_validation(epoch, valid_data, test_data, saved, verbose)
                
                if is_better:
                    wait = 0  # 破纪录了，重置忍耐值
                else:
                    wait += 1 # 没破纪录，忍耐值 +1
                    if verbose:
                        print(f"   [EarlyStopping] Patience: {wait}/{patience}")

                # 3. 触发早停
                if wait >= patience:
                    print(f"\n🛑 触发早停 (Early Stopping)！在连续 {patience} 个 Epoch 内验证集指标未提升。")
                    print(f"最佳验证集得分: {self.best_valid_score:.4f}")
                    break


        print("\n===> 训练完成 ✅")
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid





    def plot_train_loss(self, save_path=None):
        epochs = sorted(self.train_loss_dict.keys())
        losses = [self.train_loss_dict[e] for e in epochs]
        plt.plot(epochs, losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.xticks(epochs)
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.show()
