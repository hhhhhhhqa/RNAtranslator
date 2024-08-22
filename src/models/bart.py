from transformers import BartConfig, BartForConditionalGeneration

def get_bart_model(args):
    configuration = BartConfig(
                    vocab_size=args.vocab_size,
                    d_model=args.d_model,
                    max_position_embeddings = args.max_position_embeddings,
                    encoder_layers = args.encoder_layers,
                    decoder_layers = args.decoder_layers,
                    forced_eos_token_id = args.forced_eos_token_id

                    # pad_token_id=0,
                    # eos_token_id=2,
                    # decoder_start_token_id=1,
                    )
    # print(configuration)
    model = BartForConditionalGeneration(configuration).to(args.device)
    return model
