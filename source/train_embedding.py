import pandas as pd
import torch
import os
import json
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import argparse

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='embeddings')
    parser.add_argument("--model_name_or_path", default='bert-base-cased', type=str,
                        help="Path to pre-trained model or shortcut name of Huggingface model. Default: `bert-base-cased`.")
    parser.add_argument("--dataset", type=str, default='acos_drone_multi',
                    help="Dataset to use for fine-tuning the embedding. Default: `acos_drone_multi`")
    parser.add_argument("--label_name", type=str, choices=['ac', 'sp', 'ac-sp'], default='ac-sp',
                    help="Label to use for constructing sampe pairs. Default: `ac-sp`")
    parser.add_argument("--strategy", type=str, choices=['single', 'multi'], default='single',
                    help="Either using single or multi-stage fine-tuning. Default: `single`")
    parser.add_argument("--stage", type=int, default=1, help="If `strategy`=`multi`, which stage to run?. Default: 1.")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--source_scenario", type=str, default='ac_single_1')
    parser.add_argument("--margin", type=float, default=0.5, help="Hyperparam to push the negative pair at least $m$ margin apart. Default: 0.5")
    parser.add_argument("--exclude_duplicate_negative", action='store_true', help="Whether to exclude negative pair of the same sample.")
    
    args = parser.parse_args()

    args.scenario = "_".join([args.label_name, args.strategy, str(args.stage)])
    output_dir = os.path.join(args.output_dir, args.dataset, args.scenario)
    domain = args.dataset.split('_')[1]
    multiclass = None
    if domain == 'drone':
        multiclass = args.dataset.split('_')[-1]
    args.model_name = f'{domain}_{multiclass}_{args.scenario}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args.output_dir = output_dir

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    return args


def construct_dataset(args, silence=False):
    """
    Read data from file, each line is: sent####labels
    Return a pd.DataFrame that has columns: [`text`, `ac`, `sp`, `ac-sp`]
    Args:
            data_path (str): The location of the dataset in a .txt file.
            silence (bool): Whether to print the number of samples.
    Returns:
        (df, quad_df):  Pandas DataFrame(s).
    """
    data_path = os.path.join('data', args.dataset, 'train.txt')
    sentsq, acsq, spsq, acspsq = [], [], [], []
    sentss, acss, spss, acspss = [], [], [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                acs, sps= [], []

                # one sample per quad
                for tuple in eval(tuples):
                    _, ac, sp, _ = tuple
                    acsp = f'{ac}-{sp}'
                    if args.dataset.split('_')[-1] == 'data':
                        ac = ac.split('#')[-1]
                    sentsq.append(words)
                    acsq.append(ac)
                    spsq.append(sp)
                    acspsq.append(acsp)
                    acs.append(ac)
                    sps.append(sp)

                # original sample per row
                acs, sps = '-'.join(set(acs)), '-'.join(set(sps))
                acsps = f'{acs}-{sps}'
                sentss.append(words)
                acss.append(acs)
                spss.append(sps)
                acspss.append(acsps)

    if not silence:
        print(f"Total examples = {len(sentss)}")
        print(f"Total quad examples = {len(sentsq)}")
    
    dataframe_quad = pd.DataFrame({
        'text': sentsq,
        'ac': acsq,
        'sp': spsq,
        'ac-sp': acspsq
    })

    dataframe = pd.DataFrame({
        'text': sentss,
        'ac': acss,
        'sp': spss,
        'ac-sp': acspss
    })

    return dataframe, dataframe_quad 


# Create pairs for contrastive learning
def create_pairs(args, dataset: pd.DataFrame) -> list[InputExample]:
    examples = []

    if args.exclude_duplicate_negative:
        for label in dataset[args.label_name].unique():
            cluster_df = dataset[dataset[args.label_name] == label]
            other_df = dataset[dataset[args.label_name] != label]
            for i, row in cluster_df.iterrows():
                for j, other_row in cluster_df.iterrows():
                    # construct positive pairs
                    if i != j and row['text'] != other_row['text']:
                        examples.append(InputExample(texts=[row['text'], other_row['text']], label=1.0))
                for j, other_row in other_df.iterrows():
                    # construct negative pairs
                    if row['text'] != other_row['text']:
                        examples.append(InputExample(texts=[row['text'], other_row['text']], label=0.0))
    else: # include negative pairs containing exactly the same sentence or text
        for label in dataset[args.label_name].unique():
            cluster_df = dataset[dataset[args.label_name] == label]
            other_df = dataset[dataset[args.label_name] != label]
            for i, row in cluster_df.iterrows():
                for j, other_row in cluster_df.iterrows():
                    # construct positive pairs
                    if i != j:
                        examples.append(InputExample(texts=[row['text'], other_row['text']], label=1.0))
                for j, other_row in other_df.iterrows():
                    # construct negative pairs
                    examples.append(InputExample(texts=[row['text'], other_row['text']], label=0.0))
    return examples


def main():
    # initialization
    args = init_args()
    
    # Step 1: Load a pre-trained model
    if args.strategy == 'multi':
        # Load the model from the previous stage
        model_path = os.path.join('embeddings', args.dataset, args.source_scenario)
        model = SentenceTransformer(model_path)
    else:
        word_embedding_model = models.Transformer(args.model_name_or_path, do_lower_case=False)
        pooling_model = models.Pooling(word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Load your dataset
    dataframe, quad_dataframe = construct_dataset(args)
    dataframe.to_excel(os.path.join(args.output_dir, f'{args.dataset}.xlsx'), index=False)
    quad_dataframe.to_excel(os.path.join(args.output_dir, f'quad_{args.dataset}.xlsx'), index=False)

    contrastive_samples = create_pairs(args, quad_dataframe)
    # contrastive_samples.to_excel(os.path.join(args.output_dir, f'cont_{args.dataset}.xlsx'), index=False)
    # Step 3: Create DataLoader
    train_dataloader = DataLoader(contrastive_samples, shuffle=True, batch_size=args.batch_size)
    # print(train_dataloader)

    # Step 4: Define the contrastive loss
    train_loss = losses.ContrastiveLoss(model=model, margin=args.margin)

    # Optional: Define evaluator for validation
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(contrastive_samples, name=args.model_name)

    # Step 5: Train the model
    warmup_steps = int(len(train_dataloader) * args.num_epochs * 0.1)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.num_epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_dir
    )

    bert_model = next(model.modules())
    bert_model.save(path=args.output_dir, model_name=args.scenario, safe_serialization=False)
    # bert_model.save_pretrained(save_directory=args.output_dir, safe_serialization=False, variant=args.scenario)

    return 0


if __name__ == '__main__':
    main()