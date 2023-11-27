import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.Network import MYNET as Net
import numpy as np

class MYNET(Net):

    def __init__(self, args, mode=None):
        super().__init__(args,mode)


        hdim = self.num_features
        sdim = self.semantic_features
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        self.slf_attn_p = MultiHeadAttention(1, sdim, hdim, hdim, dropout=0.5)
        self.simcom = SimCom(1, hdim, hdim)


    def forward(self, input):
        if self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            support_idx, query_idx = input
            logits = self._forward(support_idx, query_idx)
            return logits

    # def cal_kl_loss(mu_poster=None, sigma_poster=None, mu_prior=None, sigma_prior=None):
    def cal_kl_loss(self, mu_poster, sigma_poster, mu_prior, sigma_prior):
        eps = 10 ** -8
        # sigma_poster = sigma_poster ** 2
        # sigma_prior = sigma_prior ** 2

        sigma_poster_matrix = sigma_poster#torch.diag_embed(sigma_poster)
        sigma_poster_matrix_det = torch.det(sigma_poster_matrix)

        sigma_prior_matrix = sigma_prior#torch.diag_embed(sigma_prior)
        sigma_prior_matrix_det = torch.det(sigma_prior_matrix)
        # if sigma_prior == 0.0:
            # sigma_prior = sigma_prior + eps
        sigma_prior_matrix_inv = torch.linalg.pinv(sigma_prior)

        #sigma_prior_matrix_inv = torch.inverse(sigma_prior)
        delta_u = (mu_prior - mu_poster).unsqueeze(1)
        delta_u_transpose = delta_u.permute(1,0)#(0, 2, 1)

        term1 = torch.sum(sigma_poster / sigma_prior, dim=1)
        # term2 = torch.bmm(
        #     torch.bmm(delta_u, sigma_prior_matrix_inv), delta_u_transpose
        # ).squeeze()
        term2 = torch.matmul(torch.matmul(delta_u_transpose, sigma_prior_matrix_inv), delta_u).squeeze()
        term3 = - mu_poster.shape[-1]
        term4 = torch.log(sigma_prior_matrix_det + eps) - torch.log(
            sigma_poster_matrix_det + eps
        )

        kl_loss = 0.5 * (term1 + term2 + term3 + term4)
        kl_loss = torch.mean(kl_loss)

        return kl_loss


    def _forward(self, support, query, semantic, is_train, var_list, cov_mix):
        emb_dim = support.size(-1)
        # get mean of the support
        proto = support.mean(dim=1)


        ways = self.args.episode_way
        proto_base = proto[:,:ways,:]
        proto_fake = proto[:,ways:,:]
        semantic_base = semantic[:,:ways,:]
        semantic_fake = semantic[:,ways:,:]


        if self.args.low_way != 0 and is_train:
            # first
            proto_low = proto[:, ways:, :]
            sim = self.simcom(proto_low).view(self.args.low_way, self.args.low_way)
            sim_sum = torch.sum(sim, dim = -1, keepdim = True).view(-1)
            _, index = sim_sum.sort(descending=True)
            # index_reserve = index[:40]
            index_reserve = index[:int(self.args.low_way/2)]

            proto_fake = proto_fake[:,index_reserve,:]
            semantic_fake = semantic_fake[:,index_reserve, :]
            # query_fake = query_fake[:,:,index_reserve,:]
            cov_mix = cov_mix[index_reserve]


            # # second, leverage proto, var_list, proto_mix, cov_mix
            final_score = []
            for ind in range(proto_fake.size(1)):
                score0 = 0.0
                for ind2 in range(proto_base.size(1)):
                    mu_poster = proto_base[:,ind2].squeeze(0) 
                    sigma_poster = var_list[ind2]#.unsqueeze(0)
                    mu_prior = proto_fake[:, ind].squeeze(0)
                    sigma_prior = cov_mix[ind]#.unsqueeze(0)
                    score0 = self.cal_kl_loss(mu_poster, sigma_poster, mu_prior, sigma_prior) + score0
                final_score.append(score0)
            final_score = torch.Tensor(final_score)
            _, index = final_score.sort(descending=True)
            index_reserve2 = index[:int(self.args.low_way/4)]
            #index_reserve2 = index[:20]

            proto_fake = proto_fake[:,index_reserve2,:]
            semantic_fake = semantic_fake[:,index_reserve2, :]
            cov_mix = cov_mix[index_reserve2]
            # query_fake = query_fake[:,:,index_reserve2,:]

            for index in range(proto_fake.size(1)):
                proto_mix0 = proto_fake.squeeze(0)[index]
                cov_mix0 = cov_mix[index]
                for num_l in range(self.args.episode_query):
                    # query_list = sampler.sample().unsqueeze(0).cuda()
                    query_list = np.random.multivariate_normal(proto_mix0.cpu().detach().numpy(), cov_mix0.cpu().detach().numpy(), None, 'ignore')
                    # # print(query_list)
                    query_list = torch.tensor(query_list, dtype=torch.float32).unsqueeze(0).cuda()#.double()
                    query_list = query_list.float()
                    if num_l == 0:
                        query_final = query_list.unsqueeze(0)
                    else:
                        query_final = torch.cat([query_final, query_list.unsqueeze(0)], dim = 0)

                proto_final = 0.0
                for num_s in range(self.args.low_shot):
                    proto_list0 = np.random.multivariate_normal(proto_mix0.cpu().detach().numpy(), cov_mix0.cpu().detach().numpy(), None, 'ignore')
                    proto_list0 = torch.tensor(proto_list0, dtype=torch.float32).unsqueeze(0).cuda()#.double()
                    proto_list0 = proto_list0.float()
                    proto_final = proto_final + proto_list0
                proto_final = proto_final/self.args.low_shot

                if index == 0:
                    # query_gen = query_final#.unsqueeze(0)
                    query_gen = query_final.unsqueeze(-2)
                    proto_list = proto_final.unsqueeze(0)
                else:
                    # query_gen = torch.cat([query_gen, query_final], 1)
                    query_gen = torch.cat([query_gen, query_final.unsqueeze(-2)], 1)
                    proto_list = torch.cat([proto_list, proto_final.unsqueeze(0)], 0)

            semantic = torch.cat([semantic_base, semantic_fake], 1)
            # proto = torch.cat([proto_base, proto_fake], 1)
            proto = torch.cat([proto_base, proto_list.squeeze(1).unsqueeze(0)], 1)
            # query = torch.cat([query_base, query_fake], 2)
            query = torch.cat([query, query_gen.unsqueeze(0).squeeze(-2)], -2)


        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = query.shape[1] * query.shape[2] #num of query*way


        # simple implementation of semantic-based branch
        if self.args.low_way != 0:
            # proto_semantic = self.slf_attn_p(semantic, proto_base, proto_base, proto_base)
            proto_semantic = self.slf_attn_p(semantic, proto, proto, proto)
            proto_update = 0.1 * proto_semantic[:,ways:,:] + 1.0 * proto[:,ways:,:]
            proto_final = torch.cat([proto[:,:ways,:], proto_update],1)
        else:
            proto_final = proto


        # combine and conduct gat for classification.
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        query = query.view(-1, emb_dim).unsqueeze(0)
        # query = query.squeeze(1).unsqueeze(0)
        combined = torch.cat([proto_final, query], 1)
        combined = self.slf_attn(combined, combined, combined, combined)

        # compute distance for all batches
        # proto_final_1, query_1 = combined.split(num_proto, 1)
        proto_final_1 = combined[:,:num_proto,:].expand(num_query, num_proto, emb_dim).contiguous()
        query_1 = combined[:,num_proto:,:].squeeze(0).unsqueeze(1)


        # query = query.view(-1, emb_dim).unsqueeze(1)

        # proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
        # proto = proto.view(num_batch*num_query, num_proto, emb_dim)

        # combined = torch.cat([proto, query], 1) # Nk x (N + 1) x d, batch_size = NK
        # combined = self.slf_attn(combined, combined, combined, combined)
        # # compute distance for all batches
        # proto, query = combined.split(num_proto, 1)


        logits = F.cosine_similarity(query_1, proto_final_1, dim=-1)
        # logits = F.cosine_similarity(query, proto, dim=-1)
        logits = logits * self.args.temperature
        # print(index_reserve)

        return logits


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        # self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        # self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        # self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.w_qs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.w_ks = nn.Linear(d_k, n_head * d_v, bias=False)
        self.w_vs = nn.Linear(d_v, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_v)

        # self.fc = nn.Linear(n_head * d_v, d_model)
        self.fc = nn.Linear(n_head * d_v, d_v)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, residual):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # residual = q
        # q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        # k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_v)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_v)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        # k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_v)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_v)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        # output, attn, log_attn = self.attention(q+residual, k+v, v+k)
        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)
        
        output = self.dropout(self.fc(output))
        # output = self.layer_norm(output + residual)
        output = self.layer_norm(output + q.view(sz_b, len_q, -1))

        return output


class SimCom(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_v, bias=False)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))


    def forward(self, q):
        d_model, d_v, n_head = self.d_model, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        temperature = np.power(d_v, 0.5)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_v)
        attn = torch.bmm(q, q.transpose(1, 2))
        attn = attn / temperature
        log_attn = F.log_softmax(attn, 2)
        log_attn = log_attn * (torch.ones_like(log_attn).cuda()-torch.eye(log_attn.size(-1)).cuda())
        return log_attn
