from transformers import GPT2Config, GPT2LMHeadModel


def get_gpt_model(args):
    # GPT2 configuration
    configuration = GPT2Config(
                vocab_size=args.vocab_size,
                n_positions=args.n_positions,
                n_embd=args.n_embd,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_inner=args.n_inner,
                activation_function=args.activation_function,
                resid_pdrop=args.resid_pdrop,
                embd_pdrop=args.embd_pdrop,
                attn_pdrop=args.attn_pdrop,
                # layer_norm_epsilon=args.layer_norm_epsilon,
                initializer_range=args.initializer_range,
                summary_type=args.summary_type,
                summary_use_proj=args.summary_use_proj,
                summary_first_dropout=args.summary_first_dropout,
                scale_attn_weights=args.scale_attn_weights,
                use_cache=args.use_cache,
                bos_token_id=args.bos_token_id,
                eos_token_id=args.eos_token_id,
                scale_attn_by_inverse_layer_idx=args.scale_attn_by_inverse_layer_idx,
                reorder_and_upcast_attn=args.reorder_and_upcast_attn
                )
    model = GPT2LMHeadModel(configuration).to(args.device)
    
    return model
