package xyz.acevedosharp;

import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;

import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.common.attributedobjects.IObjectEvaluator;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

@SuppressWarnings({"rawtypes", "unchecked"})
public class EpicEnsembleEvaluator implements IObjectEvaluator<ComponentInstance, Double> {
    private final ILabeledDataset trainSet;
    private final ILabeledDataset testSet;
    //private final List<Instances> splitInstances;

    public EpicEnsembleEvaluator(ILabeledDataset dataset, Integer seed) throws SplitFailedException, InterruptedException {
        // make split
        List<ILabeledDataset> split = SplitterUtil.getLabelStratifiedTrainTestSplit(dataset, seed, .7);

        // get train instances from split
        trainSet = split.get(0);
        testSet = split.get(1);
    }

    @Override
    public Double evaluate(ComponentInstance ensemble) {
        try {
            IWekaClassifier rawEnsemble = EpicEnsembleFactory.getEnsemble(ensemble);
            return measureAlgorithmPerformance(rawEnsemble, trainSet, testSet);
        } catch (Exception e) {
            e.printStackTrace();
            return Double.MAX_VALUE;
        }
    }

    public static Double measureAlgorithmPerformance(IWekaClassifier optimizedClassifier, ILabeledDataset<ILabeledInstance> trainSet, ILabeledDataset<ILabeledInstance> testSet) throws Exception {
        int mistakes = 0;

        Instances instances = new WekaInstances(testSet).getInstances();

        optimizedClassifier.fit(trainSet);
        //optimizedClassifier.getClassifier().buildClassifier(instances);

        for (Instance instance : instances) {
            // double value of predicted class
            double ensemblePrediction = optimizedClassifier.getClassifier().classifyInstance(instance);

            if (ensemblePrediction != instance.classValue())
                mistakes++;
        }

        Double score = mistakes * 100.0 / testSet.size();
        System.out.println("Score: " + score);
        System.out.println("===================================================");
        return score;
    }
    // 1602 -> 12226 -> 23454 -> 235461 this rising behavior only happens with larger datasets
    // 161 -> 195 -> 25 -> 344 -> 140 -> 323
}
