"""Bridge upstream video encoders to cached, LLM-ready embeddings.

The released backbone patches used a missing image2token hook. These helpers
reuse upstream multimodal preparation, including its pooling/newline behavior,
then insert the cached embeddings without re-running the vision encoder.
"""
from types import MethodType


def prepare_cached_inputs(model, input_ids, video_embeddings, attention_mask=None):
    import torch
    blocks = video_embeddings if isinstance(video_embeddings, (list, tuple)) else [video_embeddings]
    blocks = [b[0] if b.ndim == 3 and b.shape[0] == 1 else b for b in blocks]
    if any(b.ndim != 2 for b in blocks):
        raise ValueError('Cached vision embeddings must have shape [tokens, hidden_size]')
    output = []
    block_index = 0
    for row_index, row in enumerate(input_ids):
        if attention_mask is not None:
            row = row[attention_mask[row_index].bool()]
        pieces, start = [], 0
        for position in torch.where(row < 0)[0].tolist():
            if position > start:
                pieces.append(model.get_model().embed_tokens(row[start:position]))
            if block_index >= len(blocks):
                raise ValueError('Not enough cached embeddings for multimodal placeholders')
            pieces.append(blocks[block_index].to(model.device, dtype=model.get_model().embed_tokens.weight.dtype))
            block_index += 1
            start = position + 1
        if start < len(row):
            pieces.append(model.get_model().embed_tokens(row[start:]))
        output.append(torch.cat(pieces, dim=0))
    if len(output) != 1:
        raise ValueError('Main-paper serving uses batch size one')
    return output[0].unsqueeze(0)


def install_cached_video_api(model, backbone):
    import torch
    from transformers.generation.utils import GenerationMixin

    @torch.no_grad()
    def image2token(self, images, **kwargs):
        # Two ordinary sentinel tokens prevent upstream's one-token fast path.
        bos = getattr(self.config, 'bos_token_id', None) or 1
        placeholder = -200 if backbone == 'LlavaVideoQwen7b' else -201
        ids = torch.tensor([[bos, placeholder, bos]], device=self.device)
        if backbone == 'LlavaVideoQwen7b':
            prepared = self.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, images, ['video'])
            embeds = prepared[4]
        else:
            prepared = self.prepare_inputs_labels_for_multimodal(ids, None, None, None, images)
            embeds = prepared[3]
        if embeds is None:
            raise RuntimeError('Backbone did not encode the requested video')
        return [embeds[0, 1:-1].detach()]

    @torch.no_grad()
    def generate_with_video_embed(self, inputs, video_embeddings=None, **kwargs):
        attention = kwargs.pop('attention_mask', None)
        kwargs.pop('modalities', None)
        kwargs.pop('position_ids', None)
        embeddings = prepare_cached_inputs(self, inputs, video_embeddings, attention)
        limit = getattr(self.config, 'tokenizer_model_max_length', None)
        if limit and embeddings.shape[1] > limit:
            raise ValueError('Assembled multimodal prompt exceeds the context limit')
        mask = torch.ones(embeddings.shape[:2], dtype=torch.long, device=embeddings.device)
        return GenerationMixin.generate(self, inputs_embeds=embeddings, attention_mask=mask, **kwargs)

    model.image2token = MethodType(image2token, model)
    model.generate_with_video_embed = MethodType(generate_with_video_embed, model)
    return model
