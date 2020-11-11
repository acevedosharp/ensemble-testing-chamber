package xyz.acevedosharp;

import ai.libs.hasco.core.events.HASCOSolutionEvent;
import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.components.model.RefinementConfiguredSoftwareConfigurationProblem;

import java.io.*;
import java.net.URISyntaxException;
import java.sql.*;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.TimeUnit;

import ai.libs.hasco.builder.HASCOBuilder;
import ai.libs.hasco.builder.forwarddecomposition.HASCOViaFD;
import ai.libs.hasco.core.HASCOSolutionCandidate;

import ai.libs.jaicore.ml.classification.loss.dataset.EClassificationPerformanceMeasure;
import ai.libs.jaicore.ml.core.dataset.serialization.ArffDatasetAdapter;
import ai.libs.jaicore.ml.core.evaluation.evaluator.SupervisedLearnerExecutor;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import ai.libs.mlplan.weka.MLPlanWekaBuilder;

import org.api4.java.ai.ml.classification.singlelabel.evaluation.ISingleLabelClassification;
import org.api4.java.ai.ml.core.dataset.serialization.DatasetDeserializationFailedException;
import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.execution.ILearnerRunReport;
import org.api4.java.ai.ml.core.evaluation.execution.LearnerExecutionFailedException;
import org.api4.java.algorithm.Timeout;

import org.api4.java.algorithm.events.IAlgorithmEvent;
import org.api4.java.algorithm.exceptions.AlgorithmException;
import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.api4.java.common.attributedobjects.IObjectEvaluator;
import org.junit.Test;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.UnsupportedAttributeTypeException;

import java.io.File;
import java.util.stream.Collectors;

@SuppressWarnings({"unchecked", "ConstantConditions", "rawtypes"})
public class EpicPerformanceMeasurer {

    private static final List<String> DATASET_NAMES = Arrays.asList("datasets/iris.arff");
    private static final Timeout TIMEOUT = new Timeout(240, TimeUnit.SECONDS);
    private static final int REPETITIONS = 2;

    @Test
    public void epicHascoRunner() throws Exception {
        List<ILabeledDataset<?>> datasets = loadLabeledDatasets();
        for (int i = 0; i < DATASET_NAMES.size(); i++) {
            for (int j = 1; j <= REPETITIONS; j++) {
                // split dataset
                List<ILabeledDataset<?>> split = SplitterUtil.getLabelStratifiedTrainTestSplit(datasets.get(i), j, .7);

                // run the algorithm with .7 from above
                IWekaClassifier result = runHasco(split.get(0),i , j);

                // evaluate its result
                Double performance = measureAlgorithmPerformance(result, split.get(1));

                // write result to database
                writeToDatabase(new BenchmarkResult(
                        DATASET_NAMES.get(i),
                        Algos.HASCO,
                        performance,
                        j
                ));
            }
        }
    }

    @Test
    public void epicMLPlanRunner() throws SplitFailedException, InterruptedException, IOException, SQLException, LearnerExecutionFailedException, AlgorithmTimeoutedException, AlgorithmExecutionCanceledException, AlgorithmException {
        List<ILabeledDataset<?>> datasets = loadLabeledDatasets();
        for (int i = 0; i < DATASET_NAMES.size(); i++) {
            for (int j = 1; j <= REPETITIONS; j++) {
                // split dataset
                List<ILabeledDataset<?>> split = SplitterUtil.getLabelStratifiedTrainTestSplit(datasets.get(i), j, .7);

                // run the algorithm with .7 from above
                IWekaClassifier optimizedClassifier = runMLPlan(split.get(0));

                // evaluate its result with .3
                Double performance = measureAlgorithmPerformance(optimizedClassifier, split.get(1));

                // SUGGESTION: store results for every classifier and compare that against the ensemble

                // write result to database
                writeToDatabase(new BenchmarkResult(
                        DATASET_NAMES.get(i),
                        Algos.MLPLAN,
                        performance,
                        j
                ));
            }
        }
    }

    private IWekaClassifier runHasco(ILabeledDataset dataset, int i, int j) throws Exception {
        System.out.println("Execution of HASCO #"+j+" on dataset: "+ DATASET_NAMES.get(i) + " began at "+ LocalDateTime.now()+".");

        String reqInterface = "EpicEnsemble";
        File componentFile = new File(this.getClass().getClassLoader().getResource("search-space/ensemble-configuration.json").toURI());

        IObjectEvaluator<ComponentInstance, Double> evaluator = new EpicEnsembleEvaluator(dataset, j);

        RefinementConfiguredSoftwareConfigurationProblem<Double> problem =
                new RefinementConfiguredSoftwareConfigurationProblem(componentFile, reqInterface, evaluator);

        HASCOViaFD<Double> hasco = HASCOBuilder.get()
                .withProblem(problem)
                .withBestFirst()
                .withCPUs(2)
                .withTimeout(TIMEOUT)
                .withRandomCompletions()
                .withDefaultParametrizationsFirst()
                .getAlgorithm();

        ArrayList<HASCOSolutionCandidate<Double>> solutions = new ArrayList<>();

        while (hasco.hasNext()) {
            try {
                IAlgorithmEvent e = hasco.nextWithException();
                if (e instanceof HASCOSolutionEvent) {
                    @SuppressWarnings("unchecked")
                    HASCOSolutionCandidate<Double> s = ((HASCOSolutionEvent<Double>) e).getSolutionCandidate();
                    solutions.add(s);
                }
            } catch (AlgorithmTimeoutedException ex) {
                break;
            } catch (InterruptedException | AlgorithmExecutionCanceledException | AlgorithmException exx) {
                exx.printStackTrace();
            }
        }

        HASCOSolutionCandidate<Double> solution = solutions.get(0);
        for (HASCOSolutionCandidate<Double> s : solutions) {
            if (s.getScore() < solution.getScore())
                solution = s;
        }

        // transform solution into IWekaClassifier
        // very wasteful and not DRY, but it's okay since this happens outside of HASCO's runtime and doesn't affect the timeout.
        Instances instances = new WekaInstances(dataset).getInstances();
        List<IComponentInstance> nestedComponents = solution.getComponentInstance().getSatisfactionOfRequiredInterface("classifiers");
        List<Classifier> classifiers = new ArrayList<>();

        for (IComponentInstance nestedComponent : nestedComponents) {
            try {
                Classifier classifier = (Classifier) Class.forName(nestedComponent.getComponent().getName()).newInstance();
                classifier.buildClassifier(instances);
                classifiers.add(classifier);
            } catch (UnsupportedAttributeTypeException e) {
                System.out.println("Ignored component: " + nestedComponent.getComponent().getName() + " from ensemble.");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        EpicEnsemble epicEnsemble = new EpicEnsemble(classifiers, instances);

        System.out.println("Execution of HASCO ended at "+ LocalDateTime.now()+".");
        return new WekaClassifier(epicEnsemble);
    }

    private IWekaClassifier runMLPlan(ILabeledDataset dataset) throws IOException, InterruptedException, AlgorithmExecutionCanceledException, AlgorithmTimeoutedException, AlgorithmException {
        // train with the entire dataset (0.7 from runner), the split and benchmarking is handled in the runner.
        return new MLPlanWekaBuilder().withNumCpus(2).withTimeOut(TIMEOUT).withDataset(dataset).build().call();
    }

    private List<ILabeledDataset<?>> loadLabeledDatasets() {
        return DATASET_NAMES.stream().map(
                s -> {
                    try {
                        File dsFile = new File(this.getClass().getClassLoader().getResource(s).toURI());
                        return ArffDatasetAdapter.readDataset(dsFile);
                    } catch (DatasetDeserializationFailedException | URISyntaxException e) {
                        e.printStackTrace();
                    }
                    return null; // shouldn't happen
                }
        ).collect(Collectors.toList());
    }

    private Double measureAlgorithmPerformance(IWekaClassifier optimizedClassifier, ILabeledDataset dataset) throws LearnerExecutionFailedException {
        SupervisedLearnerExecutor executor = new SupervisedLearnerExecutor();
        ILearnerRunReport report = executor.execute(optimizedClassifier, dataset); // 0.3 from runner
        return EClassificationPerformanceMeasure.ERRORRATE.loss(report.getPredictionDiffList().getCastedView(Integer.class, ISingleLabelClassification.class));
    }

    @SuppressWarnings("SqlResolve")
    private void writeToDatabase(BenchmarkResult result) throws SQLException {
        Connection connection = DriverManager.getConnection(
                "jdbc:mysql://localhost:3306/epic", "root", "password"
        );
        Statement statement = connection.createStatement();

        statement.executeUpdate("insert into benchmarkresult (dataset, algo, error_rate, repetition, moment_saved, date_string) value ('" +
                result.datasetSource + "', '" + result.algorithm.toString() + "', " + result.errorRate + ", " + result.repetition + ", " + System.currentTimeMillis() + "," + LocalDateTime.now().toString() +
                ");");
    }
}
