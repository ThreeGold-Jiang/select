import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPModel
import logging
class Selector(nn.Module):
    def __init__(self, topk, selection_method='gumbel', q_dim=512, dim=512,tau=1):
        super(Selector, self).__init__()
        self.linear_Q = nn.Linear(q_dim, dim)
        self.norm_Q = nn.LayerNorm(dim, eps=1e-12)

        self.linear_K = nn.Linear(dim, dim)
        self.norm_K = nn.LayerNorm(dim, eps=1e-12)

        self.topk = topk
        self.selection_method = selection_method
        self.tau=tau

    @staticmethod
    def sample_gumbel(n, k):
        unif = torch.distributions.Uniform(0, 1).sample((n, k))
        g = -torch.log(-torch.log(unif))
        return g

    # @staticmethod
    def sample_gumbel_softmax(self, pi, temperature):
        n, k = pi.shape
        # dbg.set_trace()
        g = self.sample_gumbel(n, k).to(pi.device)
        h = (g + torch.log(pi)) / temperature
        h_max = h.max(dim=1, keepdim=True)[0]
        h = h - h_max
        cache = torch.exp(h)
        # print(pi, torch.log(pi), intmdt)
        y = cache / cache.sum(dim=-1, keepdim=True)
        return y

    def forward(self, Q, K, V):
        '''
        Q: (bs, q_dim, 1)
        K: (bs, n_select, dim), n_select could be num_obj or num_seg
        V: (bs, n_select, n_frame_per_clip, obj_num, obj_dim)
        '''
        bs, n_select, _ = K.shape
        obj_num, obj_dim = V.shape[-2:]
        # from IPython.core.debugger import set_trace;
        # set_trace()
        v_shape = V.shape
        # V = V.view(bs, n_select, -1)

        # dbg.set_trace()

        Q = self.norm_Q(self.linear_Q(Q.squeeze(dim=-1)))  # [bs, dim, 1] -> [bs, dim]
        K = self.norm_K(self.linear_K(K))  # [bs, numc, dim]
        #? logit_scale
        logit_scale = 1
        x_logits = logit_scale * K @ Q.unsqueeze(dim=-1)# [bs, numc, 1]
        x_logits = torch.softmax(x_logits.squeeze(dim=-1), dim=-1)# [bs, numc]

        selected_segs = []
        #TODO:增加hardsoftmax选择方法
        if self.selection_method=="gumbel":
            for _ in range(self.topk):
                selection_mask = F.gumbel_softmax(x_logits, tau=self.tau, dim=-1)
                if torch.isnan(selection_mask).sum() or torch.isinf(selection_mask).sum():
                    dbg.set_trace()
                selection_mask = selection_mask.unsqueeze(dim=1)
                if V.dim() == 3:
                    selected_segs.append(
                        torch.matmul(selection_mask, V.view(bs, n_select, -1)))
                else:
                    selected_segs.append(
                        torch.matmul(selection_mask, V.view(bs, n_select, -1)).view(bs, -1, obj_num, obj_dim))
        else:
            #TODO:增加hardsoftmax选择方法
            selection_indices=[]
            for _ in range(self.topk):
                selection_indice = torch.argmax(x_logits,dim=-1)
                selection_indices.append(selection_indice)
                x_logits[:,selection_indice]=0
            return selection_indices

        selected_segs = torch.cat(selected_segs, dim=1)  # [bs, topk * num_obj, CLIP_dim]

        return selected_segs
class MySelector3(nn.Module):
    def __init__(self,device,args=None,vision_dim=1024,qdim=512,hdim=1024,numc=4,nump=256,numf=16,numbeams=1):
        """
        MySelector is the same as the ISTA layer of mist

        """
        #TODO:config init
        super().__init__()
        self.dim=vision_dim
        self.hdim=hdim
        self.qdim=qdim
        self.video_proj=nn.Sequential(
            nn.Linear(self.dim,self.hdim),
            # nn.RelU()
            )
        self.seg_proj=nn.Sequential(
            nn.Linear(self.dim,self.hdim),
            # nn.RelU()
            )
        self.video_bproj=nn.Sequential(
            nn.Linear(self.hdim,self.dim),
            # nn.RelU()
            )
        self.seg_bproj = nn.Sequential(
            nn.Linear(self.hdim,self.dim),
            # nn.RelU()
            )
        self.device=device
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer= AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        for name, param in self.clip.named_parameters():
            param.requires_grad = False
        # for name, param in self.tokenizer.named_parameters():
        #     param.requires_grad = False

        self.numc=numc
        self.numf=numf//numc
        self.frames=numf
        # self.nump=256+1#(256+[CLS])
        self.nump=nump
        self.topk=2
        self.topj=128
        self.numbeams=numbeams
        if args==None:
            self.bz=3
        else:
            self.bz=args.batch_size        
        self.seg_selector = Selector(topk=self.topk,q_dim=self.qdim,dim=self.hdim,selection_method="gumbel",tau=1e-6)
        self.reg_selector = Selector(topk=self.topj,q_dim=self.qdim,dim=self.hdim,selection_method="gumbel",tau=1e-6)
    def get_clip_txt_embedding(self, question):
        bsize = question.size(0)
        # question_clip, word_clip = self.clip.encode_text(question.squeeze(dim=1))
        question_clip = self.clip.get_text_features(question.squeeze(dim=1))
        question_clip = question_clip / question_clip.norm(dim=-1, keepdim=True)   # [bsize, CLIP_dim]
        question_clip = question_clip.view(bsize, -1, 1).float()  # [bsize, 1, CLIP_dim]

        # word_clip = word_clip / word_clip.norm(dim=-1, keepdim=True)   # [bsize, num_word, CLIP_dim]
        # word_clip = word_clip.view(bsize, -1, 1).float()  # [bsize, num_word, CLIP_dim]
        return question_clip


    #TODO:增加可适配参数
    #只需patch过Qformer
    def forward(self, question_text,image_embeds,video_cls):
        ''' 
        参考 
        '''
        # def select(self):
        logging.info("SELECT DONGING!")
        print("Select donging!")
        # q_feat,  question, seg_feat, video_o
        # question\q_mask不用准备
        #TODO:video_o准备
        '''    
        video_o准备
        '''
        #TODO:根据输入对象的维度决定是否squeeze和unsqueeze
        video_dims=image_embeds.ndim
        video_in=image_embeds.contiguous()
        if video_dims==5:  #(3,1,32,256,1024)
            print(image_embeds.is_contiguous())    
            video_in=video_in.view(self.bz,self.numc*self.numf,self.nump,-1)#(bz,numc*numf,nump,dim)
            print(video_in.is_contiguous())  
            if self.hdim!=self.dim:
                video_feat=self.video_proj(video_in)#(bz,...,hdim)
                video_cls=self.seg_proj(video_cls)
            else:
                video_feat=video_in#(bz,...,hdim)
                video_cls=video_cls.contiguous()
            # video_cls=self.seg_proj(video_cls)#(bz,...,hdim)
            video_cls=video_cls.contiguous()
            video_cls_norm=video_cls/video_cls.norm(dim=-1,keepdim=True)
            video_clip = video_cls_norm.view(self.bz,self.numc,self.numf,-1)#(self.bz,numc,numf,hdim)
            seg_feat = torch.mean(video_clip, dim=-2)#(self.bz,numc,hdim)
            #TODO:seg_feat准备  
        #TODO:q_feat准备
        '''
        q_feat
        mist mean pooling+blip编码:
        question_clip = clip.tokenize(question_txt)
        '''
        #TODO:clip change to transformer
        question_clip=self.tokenizer(question_text,return_tensors="pt",padding=True)["input_ids"]
        question_clip=question_clip.to(self.device)#(self.bz,numtext)
        q_feat= self.get_clip_txt_embedding(question_clip)#(self.bz,qdim,1)
        #TODO:先实现一个视频的时候的选择
        #first choose the seg
        #TODO:选几次？
        selected_segs = self.seg_selector(q_feat, seg_feat, video_feat)# [bs, topk * numf, nump, hdim]
        q_feat_tmp = q_feat.unsqueeze(dim=1)
        # print(self.numbeams)
        q_feat_tmp = q_feat_tmp.repeat(self.numbeams, selected_segs.shape[1], 1, 1)  # [bz, topk * numf, hdim, 1]
        q_feat_tmp = q_feat_tmp.view(-1, q_feat_tmp.shape[-2], q_feat_tmp.shape[-1])  # [bz * topk * numf, hdim ,1]
        
        selected_patches = selected_segs.view(-1, selected_segs.shape[-2], selected_segs.shape[-1]) # [bz * topk * numf, nump, hdim]
        selected_patches = self.reg_selector(q_feat_tmp, selected_patches, selected_patches)  # [bs * topk * numf, topj, hdim]
        selected_patches = selected_patches.view(self.bz, -1, selected_patches.shape[-1])  # [bs, topk * numf * topj, hdim]
        if self.dim!=self.hdim:
            video_embedds=self.seg_bproj(selected_patches)# [bs, topk * numf * topj, dim]
        else:
            video_embedds=selected_patches
        if video_dims==5:
            video_embedds=video_embedds.view(self.bz,1,self.topk*self.numf,self.topj,self.dim)#(3,1,32,256,1024) 
        logging.info("SELECT DONE!")
        return video_embedds