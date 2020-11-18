package xyz.acevedosharp;

import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.ml.classification.loss.dataset.EClassificationPerformanceMeasure;
import ai.libs.jaicore.ml.core.evaluation.evaluator.SupervisedLearnerExecutor;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import org.api4.java.ai.ml.classification.singlelabel.evaluation.ISingleLabelClassification;
import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.execution.ILearnerRunReport;
import org.api4.java.ai.ml.core.evaluation.execution.LearnerExecutionFailedException;
import org.api4.java.ai.ml.core.exception.TrainingException;
import org.api4.java.common.attributedobjects.IObjectEvaluator;
import weka.core.Instances;

import java.util.*;

@SuppressWarnings({"rawtypes", "unchecked"})
public class EpicEnsembleEvaluator implements IObjectEvaluator<ComponentInstance, Double> {
    private final List<ILabeledDataset> splitILabeledDataset;
    private final List<Instances> splitInstances;

    public EpicEnsembleEvaluator(ILabeledDataset dataset, Integer seed) throws SplitFailedException, InterruptedException {

        // make split
        splitILabeledDataset = SplitterUtil.getLabelStratifiedTrainTestSplit(dataset, seed, .7);

        // get train instances from split
        Instances trainInstances = new WekaInstances(splitILabeledDataset.get(0)).getInstances();
        trainInstances.setClassIndex(trainInstances.numAttributes() - 1);

        // get test instances from split
        Instances testInstances = new WekaInstances(splitILabeledDataset.get(1)).getInstances();
        testInstances.setClassIndex(testInstances.numAttributes() - 1);

        // make new instances split
        splitInstances = Arrays.asList(trainInstances, testInstances);
    }

    @Override
    public Double evaluate(ComponentInstance ensemble) {
        try {
            IWekaClassifier epicEnsemble = EpicEnsembleFactory.getEnsemble(ensemble, splitInstances.get(0));

            double score = measureAlgorithmPerformance(epicEnsemble, splitILabeledDataset.get(1));
            System.out.println("Score: " + score);
            return score;
        } catch (Exception e) {
            e.printStackTrace();
            return Double.MAX_VALUE;
        }
    }

    public static Double measureAlgorithmPerformance(IWekaClassifier optimizedClassifier, ILabeledDataset dataset) throws LearnerExecutionFailedException, TrainingException, InterruptedException {
        long start = System.currentTimeMillis();
        optimizedClassifier.fit(dataset);

        SupervisedLearnerExecutor executor = new SupervisedLearnerExecutor();
        ILearnerRunReport report = executor.execute(optimizedClassifier, dataset); // 0.3 from runner
        double score = EClassificationPerformanceMeasure.ERRORRATE.loss(report.getPredictionDiffList().getCastedView(Integer.class, ISingleLabelClassification.class));
        System.out.println("measureAlgorithmPerformance: " + (System.currentTimeMillis() - start) + "ms");
        return score;
    }
}
