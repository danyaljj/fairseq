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
# roberta.cuda()
roberta.eval()
with open('/Users/danielk/ideaProjects/fairseq/examples/roberta/glue_data/MNLI/dev_matched.tsv') as fin:
    fin.readline()
    for index, line in tqdm(enumerate(fin)):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[8], tokens[9], tokens[15]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
        if index > 200:
            break
print('| Accuracy: ', float(ncorrect)/float(nsamples))
