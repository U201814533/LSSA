import numpy as np 
import torch
import torch.nn as nn
import random

import copy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
# import matplotlib.pyplot as plt

def show_image(image):
    # 创建一个新的图像窗口
    image=image.permute(0, 2, 3, 1)
    image=image.clone().detach().cpu().numpy()
    # 显示图像
    for i in range(image.shape[0]):
        fig, ax = plt.subplots()
        ax.imshow(image[i])
        ax.axis('off')  # 关闭坐标轴
        plt.savefig(f'./image_{i}.pdf', format='pdf', bbox_inches='tight')

class SGAttacker():
    def __init__(self, model, img_attacker, txt_attacker):
        self.model=model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker
        self.vt_sample_num = 20
    
    def attack(self, imgs, txts, txt2img, device='cpu', max_length=30, scales=None, **kwargs):
    
        bs ,_ ,_ ,_ = imgs.shape
        # original state
        # with torch.no_grad():
        #     origin_img_output = self.model.inference_image(self.img_attacker.normalization(imgs))
        #     img_supervisions = origin_img_output['image_feat'][txt2img] 
        # adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions)

        with torch.no_grad():
            txts_input = self.txt_attacker.tokenizer(txts, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt").to(device)
            txts_output = self.model.inference_text(txts_input)
            txt_supervisions = txts_output['text_feat']
        adv_imgs, ori_imgs_vt = self.img_attacker.txt_guided_attack_vt(self.model, imgs, txt2img, device, 
                                                       scales=scales, txt_embeds = txt_supervisions)
        # adv_imgs = self.img_attacker.txt_guided_attack(self.model, imgs, txt2img, device, 
        #                                                scales=scales, txt_embeds = txt_supervisions)

        with torch.no_grad():
            adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(imgs))
            img_supervisions = adv_imgs_outputs['image_feat'][txt2img]
            
            for tmp in range(self.vt_sample_num):
                ori_imgs_vt_outputs = self.model.inference_image(self.img_attacker.normalization(ori_imgs_vt[bs*tmp: bs*tmp+bs]))
                ori_adv_vt_supervisions = ori_imgs_vt_outputs['image_feat'][txt2img]
                if tmp == 0:
                    ori_img_neibor_supervisions = ori_adv_vt_supervisions
                else:
                    ori_img_neibor_supervisions = torch.cat((ori_img_neibor_supervisions, ori_adv_vt_supervisions), dim=0)

        # adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions)
        adv_txts = self.txt_attacker.img_guided_attack_multiimg(self.model, txts, img_supervisions, ori_img_neibor_supervisions)

        return adv_imgs, adv_txts

                
class ImageAttacker():
    def __init__(self, normalization, eps=2/255, steps=10, step_size=0.5/255):
        self.normalization = normalization
        self.eps = eps
        self.steps = steps 
        self.step_size = step_size 
        self.vt_sample_num = 20

    def loss_func(self, adv_imgs_embeds, txts_embeds, txt2img):  
        device = adv_imgs_embeds.device    

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)
        
        for i in range(len(txt2img)):
            it_labels[txt2img[i], i]=1
        
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos
        
        return loss
    
    def shuffle_and_merge(self, images, num_repeats=20):
        # 获取图像的高度和宽度
        height, width = images[0].shape[1], images[0].shape[2]

        # 将图像分为上下左右四个等大的部分
        half_height = height // 2
        half_width = width // 2
        top_left = images[:,:, :half_height, :half_width]
        top_right = images[:,:, :half_height, half_width:]
        bottom_left = images[:,:, half_height:, :half_width]
        bottom_right = images[:,:, half_height:, half_width:]

        shuffled_images = []  # 存储所有打乱的图像

        for i in range(num_repeats):
            # 随机打乱四个部分的顺序
            blocks = [top_left, top_right, bottom_left, bottom_right]
            random.shuffle(blocks)

            # 创建一个全零的张量，用于存储合并后的图像
            merged_image = torch.zeros_like(images)

            # 随机排列四个部分的索引
            indices = torch.randperm(4)

            # 将打乱后的四块依次填充到合并图像的对应位置
            merged_image[:,:, :half_height, :half_width] = blocks[indices[0]]
            merged_image[:,:, :half_height, half_width:] = blocks[indices[1]]
            merged_image[:,:, half_height:, :half_width] = blocks[indices[2]]
            merged_image[:,:, half_height:, half_width:] = blocks[indices[3]]
            
            shuffled_images.append(merged_image)
            
        shuffled_images=torch.cat(shuffled_images,dim=0)

        return shuffled_images
    
    
    def subshuffle_and_merge(self,images, num_repeats=5, subgrid_shuffle_index=0):
        # 获取图像的高度和宽度
        height, width = images.shape[2], images.shape[3]

        # 将图像分为上下左右四个等大的部分
        half_height = height // 2
        half_width = width // 2
        top_left = images[:,:, :half_height, :half_width]
        top_right = images[:,:, :half_height, half_width:]
        bottom_left = images[:,:, half_height:, :half_width]
        bottom_right = images[:,:, half_height:, half_width:]

        # 将指定的格子再分为等大的四个格子
        subgrid_blocks = [top_left, top_right, bottom_left, bottom_right]   # 对应j = 0, 1, 2, 3
        subgrid_block = subgrid_blocks[subgrid_shuffle_index]
        subgrid_height = subgrid_block.shape[2] // 2
        subgrid_width = subgrid_block.shape[3] // 2
        subgrid_top_left = subgrid_block[:,:, :subgrid_height, :subgrid_width]
        subgrid_top_right = subgrid_block[:,:, :subgrid_height, subgrid_width:]
        subgrid_bottom_left = subgrid_block[:,:, subgrid_height:, :subgrid_width]
        subgrid_bottom_right = subgrid_block[:,:, subgrid_height:, subgrid_width:]
        subgrid_blocks[subgrid_shuffle_index] = [subgrid_top_left, subgrid_top_right, subgrid_bottom_left, subgrid_bottom_right]

        shuffled_images = []  # 存储所有打乱的图像

        for i in range(num_repeats):
            # 随机打乱四个部分的顺序
            random.shuffle(subgrid_blocks[subgrid_shuffle_index])
            
            merged_subimage=torch.zeros_like(top_left)
            
            indices=torch.randperm(4)

            # 将打乱后的四个部分连接起来
            merged_subimage[:,:,:subgrid_height,:subgrid_width]=subgrid_blocks[subgrid_shuffle_index][indices[0]]
            merged_subimage[:,:,:subgrid_height,subgrid_width:]=subgrid_blocks[subgrid_shuffle_index][indices[1]]
            merged_subimage[:,:,subgrid_height:,:subgrid_width]=subgrid_blocks[subgrid_shuffle_index][indices[2]]
            merged_subimage[:,:,subgrid_height:,subgrid_width:]=subgrid_blocks[subgrid_shuffle_index][indices[3]]
            
            merged_image=torch.zeros_like(images)
            
            if subgrid_shuffle_index==0:
                merged_image[:,:, :half_height, half_width:] = images[:,:, :half_height, half_width:]
                merged_image[:,:, half_height:, :half_width] = images[:,:, half_height:, :half_width]
                merged_image[:,:, half_height:, half_width:] = images[:,:, half_height:, half_width:]
                merged_image[:,:, :half_height, :half_width] = merged_subimage
            if subgrid_shuffle_index==1:
                merged_image[:,:, :half_height, :half_width] = images[:,:, :half_height, :half_width]
                merged_image[:,:, half_height:, :half_width] = images[:,:, half_height:, :half_width]
                merged_image[:,:, half_height:, half_width:] = images[:,:, half_height:, half_width:]
                merged_image[:,:, :half_height, half_width:] = merged_subimage
            if subgrid_shuffle_index==2:
                merged_image[:,:, :half_height, :half_width] = images[:,:, :half_height, :half_width]
                merged_image[:,:, :half_height, half_width:] = images[:,:, :half_height, half_width:]
                merged_image[:,:, half_height:, half_width:] = images[:,:, half_height:, half_width:]
                merged_image[:,:, half_height:, :half_width] = merged_subimage
            if subgrid_shuffle_index==3:
                merged_image[:,:, :half_height, :half_width] = images[:,:, :half_height, :half_width]
                merged_image[:,:, :half_height, half_width:] = images[:,:, :half_height, half_width:]
                merged_image[:,:, half_height:, :half_width] = images[:,:, half_height:, :half_width]
                merged_image[:,:, half_height:, half_width:] = merged_subimage
                
            shuffled_images.append(merged_image)
            
        shuffled_images=torch.cat(shuffled_images,dim=0)
        # print(shuffled_images.shape)
        
        return shuffled_images
    
   
    def txt_guided_attack(self, model, imgs, txt2img, device, scales=None, txt_embeds=None):
        
        model.eval()
       
        b, _, _, _ = imgs.shape
        
        if scales is None:
            scales_num = 1
        else:
            scales_num = len(scales) +1
        postion_num=4
        sample_num=5
        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        for i in range(self.steps):
            grad=torch.zeros_like(adv_imgs)
            for _ in range(sample_num):
                adv_imgs.requires_grad_()
                scaled_imgs = self.get_scaled_imgs(adv_imgs, scales, device)
                shuffle_scaled_imgs=[]
                for j in range(postion_num):
                    subshuffle_scaled_imgs=self.subshuffle_and_merge(scaled_imgs,1,j)
                    shuffle_scaled_imgs.append(subshuffle_scaled_imgs)
                shuffle_scaled_imgs=torch.cat(shuffle_scaled_imgs,dim=0).to(device)
                
                # show_image(shuffle_scaled_imgs)
                # exit(0)
            
                # if self.normalization is not None:
                #     adv_imgs_output = model.inference_image(self.normalization(scaled_imgs))
                # else:
                #     adv_imgs_output = model.inference_image(scaled_imgs)
                
                if self.normalization is not None:
                    adv_imgs_output = model.inference_image(self.normalization(shuffle_scaled_imgs))
                else:
                    adv_imgs_output = model.inference_image(shuffle_scaled_imgs)
                    
                adv_imgs_embeds = adv_imgs_output['image_feat']
                model.zero_grad()
                with torch.enable_grad():
                    loss_list = []
                    loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                    # b*sample_num*scale_num
                    for i in range(scales_num):
                        for j in range(postion_num):
                            loss_item = self.loss_func(adv_imgs_embeds[(i*postion_num+j)*b:(i*postion_num+j)*b+b], txt_embeds, txt2img)
                            loss_list.append(loss_item.item())
                            loss += loss_item
                            
                    loss.backward()
                    grad += adv_imgs.grad 
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            # print(grad)
            
            # exit(0)
            
            perturbation = self.step_size * grad.sign()
            adv_imgs = adv_imgs.detach() + perturbation
            adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        
        return adv_imgs
    
    def txt_guided_attack_vt(self, model, imgs, txt2img, device, scales=None, txt_embeds=None):
    
        model.eval()
    
        b, _, _, _ = imgs.shape
        
        if scales is None:
            scales_num = 1
        else:
            scales_num = len(scales) +1
        postion_num=4
        sample_num=5
        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        momentum = torch.zeros_like(imgs).detach().to(device)
        decay = 1

        for i in range(self.steps):
            grad=torch.zeros_like(adv_imgs)
            for _ in range(sample_num):
                adv_imgs.requires_grad_()
                scaled_imgs = self.get_scaled_imgs(adv_imgs, scales, device)
                shuffle_scaled_imgs=[]
                for j in range(postion_num):
                    subshuffle_scaled_imgs=self.subshuffle_and_merge(scaled_imgs,1,j)
                    shuffle_scaled_imgs.append(subshuffle_scaled_imgs)
                shuffle_scaled_imgs=torch.cat(shuffle_scaled_imgs,dim=0).to(device)
            
                # if self.normalization is not None:
                #     adv_imgs_output = model.inference_image(self.normalization(scaled_imgs))
                # else:
                #     adv_imgs_output = model.inference_image(scaled_imgs)
                
                if self.normalization is not None:
                    adv_imgs_output = model.inference_image(self.normalization(shuffle_scaled_imgs))
                else:
                    adv_imgs_output = model.inference_image(shuffle_scaled_imgs)
                    
                adv_imgs_embeds = adv_imgs_output['image_feat']
                model.zero_grad()
                with torch.enable_grad():
                    loss_list = []
                    loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                    # b*sample_num*scale_num
                    for i in range(scales_num):
                        for j in range(postion_num):
                            loss_item = self.loss_func(adv_imgs_embeds[(i*postion_num+j)*b:(i*postion_num+j)*b+b], txt_embeds, txt2img)
                            loss_list.append(loss_item.item())
                            loss += loss_item
                            
                    loss.backward()
                    grad += adv_imgs.grad 
            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + decay * momentum
            momentum = grad

            perturbation = self.step_size * grad.sign()
            adv_imgs = adv_imgs.detach() + perturbation
            adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
            
        beta = 0.25
        for tmp in range(self.vt_sample_num):
            neighbor_imgs = adv_imgs.detach() + torch.randn_like(
                adv_imgs
            ).uniform_(-self.eps * beta, self.eps * beta)
            if tmp == 0:
                ori_imgs_vt = neighbor_imgs
            else:
                ori_imgs_vt = torch.cat((ori_imgs_vt, neighbor_imgs), dim=0)
        
        return adv_imgs, ori_imgs_vt


    def get_scaled_imgs(self, imgs, scales=None, device='cuda'):
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])
        
        reverse_transform = transforms.Resize(ori_shape,
                                interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio*ori_shape[0]), 
                                  int(ratio*ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                  interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)
            
            reversed_imgs = reverse_transform(scaled_imgs)
            
            result.append(reversed_imgs)
        
        return torch.cat([imgs,]+result, 0)



filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)
    

class TextAttacker():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10, threshold_pred_score=0.3, batch_size=32):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        # epsilon_txt
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls
        self.vt_sample_num = 20
        
    def Word_cos_sim(self, original_text, replace_text):
        device = self.ref_net.device
        original_text_inputs = self.tokenizer(original_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device).input_ids.squeeze()
        replace_text_inputs = self.tokenizer(replace_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device).input_ids.squeeze()
        # print(original_text_inputs)
        # print(replace_text_inputs)
        index=-1
        for i in range(len(original_text_inputs)):
            if original_text_inputs[i]!=replace_text_inputs[i]:
                index=i
                break
        
        input_embeddings=self.ref_net.get_input_embeddings()
        original_text_inputs_embeddings=input_embeddings(original_text_inputs)[index]
        replace_text_inputs_embeddings=input_embeddings(replace_text_inputs)[index]
        # print(original_text_inputs_embeddings.shape)
        
        original_text_inputs_embeddings_norm=original_text_inputs_embeddings/torch.norm(original_text_inputs_embeddings)
        replace_text_inputs_embeddings_norm=replace_text_inputs_embeddings/torch.norm(replace_text_inputs_embeddings)
        
        result=original_text_inputs_embeddings_norm*replace_text_inputs_embeddings_norm
        result=result.sum()
        # print(result)
        
        return result

    def img_guided_attack(self, net, texts, img_embeds = None):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        # input_embeddings=net.text_encoder.get_input_embeddings()
        # print(text_inputs.input_ids)
        # print(input_embeddings(text_inputs.input_ids).shape)
        # exit(0)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()

        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)


                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    
                    # if self.Word_cos_sim(text,' '.join(temp_replace))<0.4:
                    #     continue
                    
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_feat'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_feat'].flatten(1)

                loss = self.loss_func(replace_embeds, img_embeds, i)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse
    
    def img_guided_attack_multiimg(self, net, texts, img_embeds = None, ori_img_neibor_embeds = None):
        device = self.ref_net.device

        bs, _, = img_embeds.shape

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()

        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                                self.threshold_pred_score)


                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_feat'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_feat'].flatten(1)

                # ori_img_neibor_embeds
                loss_ori = self.loss_func(replace_embeds, img_embeds, i)
                loss_vt = 0
                for loss_i in range(self.vt_sample_num):
                    loss_vt += self.loss_func(replace_embeds, ori_img_neibor_embeds[bs*loss_i:bs*loss_i+bs], i)
                    
                loss = 0.5 * loss_ori + 0.5 * (loss_vt / self.vt_sample_num)

                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse
        

    def loss_func(self, txt_embeds, img_embeds, label):
        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1) 
        loss = loss_TaIcpos
        return loss


    def attack(self, net, texts):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_embed'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_embed'].flatten(1).detach()



        criterion = torch.nn.KLDivLoss(reduction='none')
        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)


                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_embed'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_embed'].flatten(1)

                loss = criterion(replace_embeds.log_softmax(dim=-1), origin_embeds[i].softmax(dim=-1).repeat(len(replace_embeds), 1))
                
                loss = loss.sum(dim=-1)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

 
    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i+batch_size], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1), origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))

        return import_scores.sum(dim=-1)



def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words
