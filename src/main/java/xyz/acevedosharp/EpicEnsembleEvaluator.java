package xyz.acevedosharp;

import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.common.attributedobjects.IObjectEvaluator;
import weka.classifiers.Classifier;
import ai.libs.jaicore.ml.weka.WekaUtil;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.UnsupportedAttributeTypeException;

import java.util.*;

@SuppressWarnings({"rawtypes", "unchecked"})
public class EpicEnsembleEvaluator implements IObjectEvaluator<ComponentInstance, Double> {
    private final List<Instances> split;

    public EpicEnsembleEvaluator(ILabeledDataset dataset, Integer seed) throws SplitFailedException, InterruptedException {
        // here it doesn't transform the dataset for every evaluation.
        Instances instances = new WekaInstances(dataset).getInstances();
        instances.setClassIndex(instances.numAttributes() - 1);
        split = WekaUtil.getStratifiedSplit(instances, seed, .7);
    }

    @Override
    public Double evaluate(ComponentInstance ensemble) {
        List<IComponentInstance> nestedComponents = ensemble.getSatisfactionOfRequiredInterface("classifiers");
        List<Classifier> classifiers = new ArrayList<>();

        for (IComponentInstance nestedComponent : nestedComponents) {
            try {
                Classifier classifier = (Classifier) Class.forName(nestedComponent.getComponent().getName()).newInstance();
                classifier.buildClassifier(split.get(0));
                classifiers.add(classifier);
            } catch (UnsupportedAttributeTypeException e) {
                // do nothing. This is for skipping classifiers that don't work on the dataset.
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        EpicEnsemble epicEnsemble = null;
        try {
            epicEnsemble = new EpicEnsemble(classifiers, split.get(0));
        } catch (Exception e) {
            e.printStackTrace();
        }

        int mistakes = 0;
        // make ensemble prediction for every instance in instances
        for (Instance instance : split.get(1)) {
            double ensemblePrediction = 0;
            try {
                ensemblePrediction = epicEnsemble.classifyInstance(instance);
            } catch (Exception e) {
                e.printStackTrace();
            }
            if (ensemblePrediction != instance.classValue()) {
                mistakes++;
            }
        }
        double score = mistakes * 100.0 / split.get(1).size();
        for (int i = 0; i < epicEnsemble.getClassifiers().length; i++) {
            System.out.println(epicEnsemble.getClassifiers()[i].getClass());
        }
        System.out.println("Considered one ensemble with score: " + score);
        return score;
    }
}
