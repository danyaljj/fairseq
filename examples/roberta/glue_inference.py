from tqdm import tqdm

from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    '/Users/danielk/ideaProjects/fairseq/examples/roberta/mnli-checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='MNLI-bin'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.target_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open('/Users/danielk/ideaProjects/fairseq/examples/roberta/glue_data/MNLI/dev_matched.tsv') as fin:
    fin.readline()
    with open('glue_data/MNLI/dev_matched_predictions.tsv', 'w+') as fout:
        for index, line in tqdm(enumerate(fin)):
            tokens = line.strip().split('\t')
            sent1, sent2, target = tokens[8], tokens[9], tokens[-1]
            tokens = roberta.encode(sent1, sent2)
            prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
            prediction_label = label_fn(prediction)
            line2 = line.replace("\n", '')
            fout.write(f"{tokens[0]}\t{sent1}\t{sent2}\t{target}\t{prediction_label}\n")
            ncorrect += int(prediction_label == target)
            nsamples += 1
            # if index > 200:
            #     break
            if index % 100 == 0 and nsamples > 0:
                print('| Accuracy: ', float(ncorrect) / float(nsamples))

print('| Accuracy: ', float(ncorrect)/float(nsamples))
