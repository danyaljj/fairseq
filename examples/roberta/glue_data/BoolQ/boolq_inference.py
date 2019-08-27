from tqdm import tqdm

from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    '../../boolq-checkpoint/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='../BoolQ-bin'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.target_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open('dev.tsv') as fin:
    fin.readline()
    with open('predictions/dev_predictions.tsv', 'w+') as fout:
        for index, line in tqdm(enumerate(fin)):
            tokens = line.strip().split('\t')
            sent1, sent2, target = tokens[1], tokens[2], tokens[3]
            if len(sent1) > 510:
                sent1 = sent1[:512]
            tokens = roberta.encode(sent1, sent2)
            prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
            prediction_label = label_fn(prediction)
            line2 = line.replace("\n", '')
            fout.write(f"{sent1}\t{sent2}\t{target}\t{prediction_label}\n")
            ncorrect += int(prediction_label == target)
            nsamples += 1
            # if index > 200:
            #     break
            if index % 100 == 0 and nsamples > 0:
                print('| Accuracy: ', float(ncorrect) / float(nsamples))

print('| Accuracy: ', float(ncorrect)/float(nsamples))
