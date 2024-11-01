from transformers import T5Config, T5ForConditionalGeneration

def get_t5_model(args):
    configuration = T5Config(
                    vocab_size=args.vocab_size,
                    d_model=args.d_model,
                    d_kv=args.d_kv,
                    d_ff=args.d_ff,
                    num_layers=args.num_layers,
                    num_decoder_layers=args.num_decoder_layers,
                    num_heads=args.num_heads,
                    relative_attention_num_buckets=args.relative_attention_num_buckets,
                    relative_attention_max_distance=args.relative_attention_max_distance,
                    dropout_rate=args.dropout_rate,
                    classifier_dropout=args.classifier_dropout,
                    initializer_factor=args.initializer_factor,
                    feed_forward_proj=args.feed_forward_proj,
                    use_cache=args.use_cache,
                    
                    bos_token_id=1, 
                    decoder_start_token_id=0,
                    )

# What worked for protein RNA
    # configuration = T5Config(
    #         vocab_size=1000,
    #         bos_token_id=1,
    #         d_model= 512,
    #         num_heads = 12,
    #         d_kv=64,
    #         d_ff = 1024,
    #         num_layers = 6,
    #         decoder_start_token_id=0)
                                                
    model = T5ForConditionalGeneration(configuration).to(args.device)
    return model
