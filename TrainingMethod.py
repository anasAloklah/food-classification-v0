from sklearn.model_selection import StratifiedKFold
import numpy
def sampleTraining(model,train_generator,validation_generator,batch_size,epochs):

    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_generator.classes) // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=len(validation_generator.classes) // batch_size, shuffle=True)

def KFoldCrossValidtionTrainig(model,dataset,batch_size,epochs):

    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # load pima indians dataset
    dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    XX=dataset._
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X, Y):
        # Fit the model
        model.fit(X[train], Y[train], epochs=epochs, batch_size=batch_size, verbose=0)
        # evaluate the model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)