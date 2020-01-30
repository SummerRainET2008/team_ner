#coding: utf8

# compute f1 score is modified from conlleval.pl
def __startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart=False):
  if prevTag == 'B' and tag == 'B':
    chunkStart = True
  if prevTag == 'I' and tag == 'B':
    chunkStart = True
  if prevTag == 'O' and tag == 'B':
    chunkStart = True
  if tag != 'O' and tag != '.' and prevTagType != tagType:
    chunkStart = True
  return chunkStart

def __endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd=False):
  if prevTag == 'B' and tag == 'B':
    chunkEnd = True
  if prevTag == 'B' and tag == 'O':
    chunkEnd = True
  if prevTag == 'I' and tag == 'B':
    chunkEnd = True
  if prevTag == 'I' and tag == 'O':
    chunkEnd = True
  if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
    chunkEnd = True
  return chunkEnd

def __splitTagType(tag):
  s = tag.split('-')
  if len(s) > 2 or len(s) == 0:
    raise ValueError('tag format wrong. it must be B-xxx.xxx')
  if len(s) == 1:
    tag = s[0]
    tagType = ""
  else:
    tag = s[0]
    tagType = s[1]
  return tag, tagType

def compute_f1_score(correct_slots, pred_slots):
  '''
  Args:
    correct_slots: [[],[],[]]
    pred_slots: [[],[],[]]
  '''
  correctChunkCnt = 0
  foundCorrectCnt = 0
  foundPredCnt = 0
  for correct_slot, pred_slot in zip(correct_slots, pred_slots):
    correct = False
    lastCorrectTag = 'O'  # o not 0
    lastCorrectType = ''
    lastPredTag = 'O'
    lastPredType = ''

    for c, p in zip(correct_slot, pred_slot):
      correctTag, correctType = __splitTagType(c)
      predTag, predType = __splitTagType(p)

      if correct == True:  # until we can be sure it is correct,it's incorrect
        if __endOfChunk(
          lastCorrectTag, correctTag, lastCorrectType, correctType
          ) == True and __endOfChunk(
          lastPredTag, predTag, lastPredType, predType
          ) == True and (lastCorrectType == lastPredType):
          # we can finally say it's correct and correctChunkCnt += 1
          correctChunkCnt += 1
          correct = False # for next iteration

        elif __endOfChunk(
          lastCorrectTag, correctTag, lastCorrectType, correctType
          ) != __endOfChunk(
          lastPredTag, predTag, lastPredType, predType) or (
          correctType != predType):
          correct = False

      if __startOfChunk(
        lastCorrectTag, correctTag, lastCorrectType, correctType
        ) == True and __startOfChunk(
        lastPredTag, predTag, lastPredType, predType
        ) == True and (
        correctType == predType):
        correct = True

      if __startOfChunk(
          lastCorrectTag, correctTag, lastCorrectType, correctType
         ) == True:
        foundCorrectCnt += 1

      if __startOfChunk(lastPredTag, predTag, lastPredType, predType):
        foundPredCnt += 1

      lastCorrectTag = correctTag
      lastCorrectType = correctType
      lastPredTag = predTag
      lastPredType = predType

    if correct == True:
      correctChunkCnt += 1
    else:
      pass

  if foundPredCnt > 0:
    precision = 100 * correctChunkCnt / foundPredCnt
  else:
    precision = 0

  if foundCorrectCnt > 0:
    recall = 100 * correctChunkCnt / foundCorrectCnt
  else:
    recall = 0
  if (precision + recall) > 0:
    f1 = (2 * precision * recall) / (precision + recall)
  else:
    f1 = 0

  return f1, precision, recall

def get_sent_acc(truth_file, pred_file):
  n_total = 0
  n_correct = 0
  intent_correct = 0
  with open(truth_file) as f_truth, open(pred_file) as f_pred:
    for i, (truth, pred) in enumerate(zip(f_truth, f_pred)):
      n_total += 1
      if pred.lower() == truth.lower():
        n_correct += 1
      if pred.split()[0].lower() == truth.split()[0].lower():
        intent_correct+=1
  try:
    acc = (n_correct / n_total) * 100
    intent_acc = (intent_correct/n_total)*100
  except:
    acc = 0
    intent_acc = 0

  return intent_acc, acc
