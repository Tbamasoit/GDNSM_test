import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from time import time
from logging import getLogger
from GDNSM.utils.topk_evaluator import TopKEvaluator
from GDNSM.utils.utils import early_stopping, dict2str
import matplotlib.pyplot as plt

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
        self.learner = getattr(config, 'learner', 'adam')
        self.learning_rate = getattr(config, 'learning_rate', 0.001)
        self.epochs = getattr(config, 'epochs', 100)
        self.eval_step = min(getattr(config, 'eval_step', 1), self.epochs)
        self.stopping_step = getattr(config, 'stopping_step', 10)
        self.clip_grad_norm = getattr(config, 'clip_grad_norm', None)
        self.valid_metric = getattr(config, 'valid_metric', 'NDCG@20').lower()
        self.valid_metric_bigger = getattr(config, 'valid_metric_bigger', True)
        self.test_batch_size = getattr(config, 'eval_batch_size', 128)
        self.device = getattr(config, 'device', 'cpu')
        self.weight_decay = getattr(config, 'weight_decay', 0.0)
        if isinstance(self.weight_decay, str):
            self.weight_decay = eval(self.weight_decay)

        self.req_training = getattr(config, 'req_training', True)
        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {f'{j.lower()}@{k}': 0.0 for j, k in itertools.product(
            getattr(config, 'metrics', ['NDCG']), getattr(config, 'topk', [20])
        )}
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        lr_scheduler = getattr(config, 'learning_rate_scheduler', [0.96, 50])
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)

        self.eval_type = getattr(config, 'eval_type', 'full')
        self.evaluator = TopKEvaluator(config)

        # GDNSM 多目标训练参数
        self.alpha1 = getattr(config, 'alpha1', 1.0)
        self.alpha2 = getattr(config, 'alpha2', 1.0)
        self.beta = getattr(config, 'beta', 1)

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

    def _train_epoch(self, train_data):
        self.model.train()
        total_loss = 0.0

        for batch in train_data:
            # batch 可能只有 users 和 pos_items
            users = batch[0]
            pos_items = batch[1]

            # 如果 batch[2] 不存在，则生成随机负样本
            if len(batch) >= 3:
                neg_items = batch[2]
            else:
                num_items = self.model.n_items
                neg_items = torch.randint(0, num_items, pos_items.shape, device=pos_items.device)

            # 获取用户、正负物品 embedding
            ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.model.forward(self.model.norm_adj, train=True)
            u_g_embeddings = ua_embeddings[users]
            pos_i_g_embeddings = ia_embeddings[pos_items]
            neg_i_g_embeddings = ia_embeddings[neg_items]

            # BPR loss
            batch_mf_loss, batch_emb_loss, batch_reg_loss = self.model.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

            # 对比学习 loss
            side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.model.n_users, self.model.n_items], dim=0)
            content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.model.n_users, self.model.n_items], dim=0)
            cl_loss = self.model.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + \
                    self.model.InfoNCE(side_embeds_users[users], content_embeds_user[users], 0.2)

            # Diffusion loss
            # 构建 item embedding 作为 diffusion 输入
            x_input = ia_embeddings[pos_items]  # 正样本 item embedding
            labels = torch.cat([u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings], dim=1)  # cond embedding
            diffusion_loss = self.model.diffusion_MM(x_input, labels, device=x_input.device)

            # 总 loss
            loss = batch_mf_loss + batch_emb_loss + batch_reg_loss + self.model.cl_loss * cl_loss + diffusion_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_data)


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False):
        self.model.eval()
        batch_matrix_list = []
        for batch in eval_data:
            scores = self.model.full_sort_predict(batch)
            masked_items = batch[1]
            scores[masked_items[0], masked_items[1]] = -1e10
            _, topk_index = torch.topk(scores, max(getattr(self.config, 'topk', [20])), dim=-1)
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test)

    def fit(self, train_data, valid_data=None, test_data=None, verbose=True, saved=True):
        for epoch in range(self.epochs):
            start_time = time()
            train_loss = self._train_epoch(train_data)
            self.lr_scheduler.step()
            self.train_loss_dict[epoch] = train_loss
            end_time = time()
            
            if verbose:
                self.logger.info(f'Epoch {epoch} | Train Loss: {train_loss:.4f} | Time: {end_time - start_time:.2f}s')

            # 验证逻辑
            if valid_data and (epoch + 1) % self.eval_step == 0:
                valid_result = self.evaluate(valid_data)
                valid_score = valid_result.get(self.valid_metric, valid_result.get('NDCG@20', 0.0))
                
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger
                )

                test_result = self.evaluate(test_data, is_test=True) if test_data else None

                if update_flag:
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result

                    # 保存模型
                    if saved:
                        save_path = f'best_model_{getattr(self.config, "model", "GDNSM")}.pth'
                        torch.save(self.model.state_dict(), save_path)
                        self.logger.info(f'Best model saved to {save_path}')

                if verbose:
                    self.logger.info(f'Valid Score: {valid_score:.4f}')
                    if test_result:
                        self.logger.info(f'Test Result: \n{dict2str(test_result)}')

                if stop_flag:
                    if verbose:
                        self.logger.info(f'Early stopping at epoch {epoch}')
                    break

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
