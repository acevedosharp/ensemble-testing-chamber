package xyz.acevedosharp;

import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.common.attributedobjects.IObjectEvaluator;
import org.api4.java.common.attributedobjects.ObjectEvaluationFailedException;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import ai.libs.jaicore.ml.weka.WekaUtil;
import weka.core.Instances;

import java.io.FileReader;
import java.util.List;
import java.util.Random;

public class EpicCompositionEvaluator implements IObjectEvaluator<ComponentInstance, Double> {
    private final ILabeledDataset<?> ds;

    public EpicCompositionEvaluator(ILabeledDataset<?> ds) {
        this.ds = ds;
    }

    @Override
    public Double evaluate(ComponentInstance object) throws InterruptedException, ObjectEvaluationFailedException {
        // split dataset 70/30
        try {
            List<Instances> split = WekaUtil.getStratifiedSplit(new Instances(new FileReader(ds)), 42, .7);  //TODO: parametrize seed

            Classifier classifier = (Classifier) Class.forName(object.getComponent().getName()).newInstance();
            classifier.buildClassifier((Instances) split.get(0));
            System.out.println(classifier);
        } catch (Exception e) {
            e.printStackTrace();
        }


        // train on 70

        // evaluate on 30
        return 5.6;
    }
}
