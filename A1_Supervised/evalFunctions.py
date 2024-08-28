import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    tot = len(LPred)
    correct = 0
    for i in range(tot):
        if LPred[i] == LTrue[i]:
            correct +=1
    acc = correct / tot
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """
    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    tot = len(LTrue)
    classes = np.unique(LTrue)
    nClasses = len(classes)
    #print(classes)
    #print(nClasses)

    cM = np.zeros((nClasses,nClasses))
    for i in range(tot):
        cM[LPred[i].astype(int),LTrue[i].astype(int)] += 1
    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    acc = np.sum(np.diag(cM))/np.sum(cM)
    # ============================================
    
    return acc
