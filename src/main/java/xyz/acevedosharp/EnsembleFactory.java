package xyz.acevedosharp;

import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.components.model.ComponentInstance;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.Vote;

import java.util.ArrayList;
import java.util.List;

public class EnsembleFactory {

    public Classifier getEnsemble(ComponentInstance ensemble) {
        List<IComponentInstance> nestedComponents = ensemble.getSatisfactionOfRequiredInterface("EpicEnsemble");
        List<Classifier> classifiers = new ArrayList<>();

        for (IComponentInstance component : nestedComponents) {
            // create a classifier from name

            // prepare its arguments

            //
            new NaiveBayes().getOptions();
        }
        return new Vote();
    }
}
