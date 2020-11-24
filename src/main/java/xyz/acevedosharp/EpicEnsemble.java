package xyz.acevedosharp;

import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.UnsupportedAttributeTypeException;

import java.util.*;

@SuppressWarnings("ConstantConditions")
public class EpicEnsemble implements Classifier {
    private final List<Classifier> classifiers;

    // Only use already built classifiers
    public EpicEnsemble(List<Classifier> classifiers) {
        this.classifiers = classifiers;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        long start = System.currentTimeMillis();

        if (Thread.interrupted()) {
            throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
        }

        List<Integer> delList = new ArrayList<>();

        // build every classifier in composition and not include ones that don't support problem type
        for (int i = classifiers.size() - 1; i >= 0; i--) { // iterate backwards so that no need to sort delList
            try {
                if (Thread.interrupted()) {
                    throw new InterruptedException("Thread got interrupted, thus, kill WEKA.");
                }
                classifiers.get(i).buildClassifier(data);
                System.out.println("Building: " + classifiers.get(i).getClass().getName());
            } catch (UnsupportedAttributeTypeException e) {
                delList.add(i);
            }
        }

        // remove from classifiers those who could not be built
        for (Integer index : delList) {
            classifiers.remove((int) index);
        }

        System.out.println("Resulting ensemble: ");
        for (int i = 0; i < classifiers.size(); i++) {
            System.out.println("\t" + i + ") " + classifiers.get(i).getClass().getName());
        }
        System.out.println("Building it took " + (System.currentTimeMillis() - start) + "ms");
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        Map<Double, Integer> classPredictions = new HashMap<>();

        for (Classifier classifier : classifiers) {
            Double prediction = classifier.classifyInstance(instance);
            if (classPredictions.containsKey(prediction)) {
                classPredictions.compute(prediction, (aDouble, integer) -> integer + 1);
            } else {
                classPredictions.put(prediction, 1);
            }
        }

        // find most frequent prediction and use it as the ensemble's prediction (hard voting)
        Double ensemblePrediction = classPredictions.entrySet().iterator().next().getKey(); // first key by default
        for (Map.Entry<Double, Integer> entry : classPredictions.entrySet()) {
            if (!entry.getKey().equals(ensemblePrediction))
                if (entry.getValue() > classPredictions.get(ensemblePrediction))
                    ensemblePrediction = entry.getKey();
        }
        return ensemblePrediction;
    }

    // we're not going to use this
    @Override
    public double[] distributionForInstance(Instance instance) {
        return null;
    }

    // we're not going to use this
    @Override
    public Capabilities getCapabilities() {
        return null;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        classifiers.forEach(
                classifier -> sb
                        .append(classifier.getClass().getName())
                        .append(", ")
        );

        sb.deleteCharAt(sb.length()-1);
        sb.deleteCharAt(sb.length()-1);

        return sb.toString();
    }
}
