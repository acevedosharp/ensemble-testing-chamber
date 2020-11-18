package xyz.acevedosharp;

import weka.classifiers.Classifier;
import weka.classifiers.rules.M5Rules;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class EpicEnsemble implements Classifier {
    private final List<Classifier> classifiers;

    // Only use already built classifiers
    public EpicEnsemble(List<Classifier> classifiers) {
        this.classifiers = classifiers;
    }

    @Override
    public void buildClassifier(Instances data) {
        List<Integer> delList = new ArrayList<>();

        for (int i = classifiers.size() - 1; i != 0; i--) { // iterate backwards so that no need to sort delList
            try {
                classifiers.get(i).buildClassifier(data);
            } catch (Exception e) {
                System.out.println("Ignored " + classifiers.get(i).getClass().toString());
                delList.add(i);
            }
        }

        for (Integer index : delList) {
            classifiers.remove((int) index);
        }
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
        Double ensemblePrediction = classPredictions.entrySet().iterator().next().getKey(); // first key
        for (Map.Entry<Double, Integer> entry : classPredictions.entrySet()) {
            if (!entry.getKey().equals(ensemblePrediction))
                if (entry.getValue() > classPredictions.get(ensemblePrediction))
                    ensemblePrediction = entry.getKey();
        }

        return ensemblePrediction;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] doubles = new double[instance.numClasses()];

        for (Classifier classifier : classifiers) {
            double[] dist = null;
            try {
                 dist = classifier.distributionForInstance(instance);
            } catch (Exception e) {
                e.printStackTrace();
            }
            for (int i = 0; i < dist.length; i++) {
                doubles[i] += dist[i];
            }
        }

        for (int i = 0; i < doubles.length; i++) {
            doubles[i] /= instance.numClasses();
        }

        return doubles;
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }
}
