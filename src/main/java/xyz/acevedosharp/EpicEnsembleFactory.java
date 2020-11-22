package xyz.acevedosharp;

import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import weka.classifiers.Classifier;
import weka.core.OptionHandler;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@SuppressWarnings("StatementWithEmptyBody")
public class EpicEnsembleFactory {
    public static IWekaClassifier getEnsemble(ComponentInstance ensemble) {
        long start = System.currentTimeMillis();

        List<IComponentInstance> nestedComponents = ensemble.getSatisfactionOfRequiredInterface("classifiers");
        List<Classifier> classifiers = new ArrayList<>();

        for (IComponentInstance component : nestedComponents) {
            try {
                // create a classifier from name
                Classifier classifier = (Classifier) Class.forName(component.getComponent().getName()).newInstance();

                // prepare its arguments
                List<String> parameters = new ArrayList<>();
                for (Map.Entry<String, String> pair : component.getParameterValues().entrySet()) {
                    // exclude parameters whose name is not a one character flag
                    if (pair.getKey().length() == 1) {
                        String value = pair.getValue();
                        if (value.equals("true")) {
                            parameters.add("-" + pair.getKey());
                        }
                        else if (value.equals("false")) {} // do nothing
                        else {// normall parameter (not boolean)
                            parameters.add("-" + pair.getKey());
                            parameters.add(value);
                        }
                    }
                }
                ((OptionHandler) classifier).setOptions(parameters.toArray(new String[0]));

                classifiers.add(classifier);
                System.out.println("loaded classifier " + classifier.getClass() + " with params " + parameters);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        EpicEnsemble epicEnsemble = new EpicEnsemble(classifiers);
        System.out.println("GetEnsemble: " + (System.currentTimeMillis() - start) + "ms");
        return new WekaClassifier(epicEnsemble);
    }
}
