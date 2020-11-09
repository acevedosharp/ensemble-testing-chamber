package xyz.acevedosharp;

import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.components.model.ComponentInstance;
import org.api4.java.common.attributedobjects.IObjectEvaluator;
import weka.classifiers.Classifier;
import ai.libs.jaicore.ml.weka.WekaUtil;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.UnsupportedAttributeTypeException;

import java.io.File;
import java.io.FileReader;
import java.util.*;

public class EpicEnsembleEvaluator implements IObjectEvaluator<ComponentInstance, Double> {
    private final File ds;

    public EpicEnsembleEvaluator(File ds) {
        this.ds = ds;
    }

    @Override
    public Double evaluate(ComponentInstance ensemble) {
        System.out.println("================================================");
        try {
            Instances dataset = new Instances(new FileReader(ds));
            dataset.setClassIndex(dataset.numAttributes() - 1);
            List<Instances> split = WekaUtil.getStratifiedSplit(dataset, new Random().nextLong(), .7);

            List<IComponentInstance> nestedComponents = ensemble.getSatisfactionOfRequiredInterface("classifiers");
            System.out.println("Ensemble with " + nestedComponents.size() + " nested components: ");
            nestedComponents.forEach(iComponentInstance -> System.out.println("\t" + iComponentInstance.getComponent().getName()));
            List<Classifier> classifiers = new ArrayList<>();

            for (IComponentInstance nestedComponent : nestedComponents) {
                try {
                    Classifier classifier = (Classifier) Class.forName(nestedComponent.getComponent().getName()).newInstance();
                    classifier.buildClassifier(split.get(0));
                    classifiers.add(classifier);
                } catch (UnsupportedAttributeTypeException e) {
                    System.out.println("Ignored component: " + nestedComponent.getComponent().getName() + " from ensemble.");
                }
            }

            int mistakes = 0;
            // make ensemble prediction for every instance in instances
            for (Instance instance : split.get(1)) {
                Map<Double, Integer> classPredictions = new HashMap<>();
                // predict with every nested component, reminder that if a nested component throws an exception, the whole ensemble is discarded (instead of that particular component)
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

                if (ensemblePrediction != instance.classValue()) {
                    mistakes++;
                }
            }
            Double score = mistakes * 100.0 / split.get(1).size();
            System.out.println("Considered ensemble with score " + score);
            return score;
        } catch (Exception e) {
            System.out.println("Ignored ensemble with message " + e.getMessage());
        }
        return Double.MAX_VALUE; // disqualify ensemble if something weird happens...
    }
}
