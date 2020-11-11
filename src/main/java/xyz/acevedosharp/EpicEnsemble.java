package xyz.acevedosharp;

import weka.classifiers.Classifier;
import weka.classifiers.meta.Vote;
import weka.core.Instances;

import java.util.List;

public class EpicEnsemble extends Vote {
    public EpicEnsemble(List<Classifier> nestedClassifiers, Instances instances) throws Exception {
        this.m_Classifiers = nestedClassifiers.toArray(new Classifier[0]);
        this.buildClassifier(instances);
    }
}
