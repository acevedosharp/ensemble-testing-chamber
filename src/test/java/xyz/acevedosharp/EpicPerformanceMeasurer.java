package xyz.acevedosharp;

import ai.libs.hasco.core.events.HASCOSolutionEvent;
import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.components.model.RefinementConfiguredSoftwareConfigurationProblem;

import java.io.*;
import java.net.URISyntaxException;
import java.sql.*;
import java.time.Instant;
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
import ai.libs.mlplan.core.MLPlan;
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

import java.io.File;
import java.util.stream.Collectors;

@SuppressWarnings({"UnusedLabel", "unchecked", "ConstantConditions", "rawtypes", "ForLoopReplaceableByForEach"})
public class EpicPerformanceMeasurer {

    private static final List<String> DATASET_NAMES = Arrays.asList("datasets/iris.arff");
    private static final Timeout TIMEOUT = new Timeout(10, TimeUnit.SECONDS);

    @Test
    public void epicHascoRunner() throws IOException, URISyntaxException, SQLException {
        List<File> datasets = datasetsAsFiles();

        for (int i = 0; i < datasets.size(); i++) {
            for (int j = 1; j <= 5; j++) {
                runHasco(datasets.get(i), i, j);
            }
        }
    }

    @Test
    public void epicMLPlanRunner() throws SplitFailedException, URISyntaxException, DatasetDeserializationFailedException, InterruptedException, IOException {
        List<ILabeledDataset<?>> datasets = labeledDatasets();

        for (int i = 0; i < datasets.size(); i++) {
            for (int j = 1; j <= 5; j++) {
                runMLPlan(datasets.get(i), i, j);
            }
        }
    }

    @Test
    public void testDb() throws SQLException {
        writeToDatabase(new BenchmarkResult("monke", Algos.HASCO, 10.4, 1));
    }

    @SuppressWarnings("SqlResolve")
    private void writeToDatabase(BenchmarkResult result) throws SQLException {
        Connection connection = DriverManager.getConnection(
                "jdbc:mysql://localhost:3306/epic", "root", "password"
        );
        Statement statement = connection.createStatement();

        int res = statement.executeUpdate("insert into benchmarkresult (dataset, algo, error_rate, repetition, moment_saved) value ('" +
                result.datasetSource + "', '" + result.algorithm.toString() + "', " + result.errorRate + ", " + result.repetition + ", " + System.currentTimeMillis() +
                ");");
        System.out.println(res);
    }

    private void runHasco(File dsFile, int i, int j) throws URISyntaxException, IOException, SQLException {
        String reqInterface = "EpicEnsemble";
        File componentFile = new File(this.getClass().getClassLoader().getResource("search-space/ensemble-configuration.json").toURI());

        IObjectEvaluator<ComponentInstance, Double> evaluator = new EpicEnsembleEvaluator(dsFile);

        RefinementConfiguredSoftwareConfigurationProblem<Double> problem =
                new RefinementConfiguredSoftwareConfigurationProblem(componentFile, reqInterface, evaluator);

        HASCOViaFD<Double> hasco = HASCOBuilder.get()
                .withProblem(problem)
                .withBestFirst()
                .withRandomCompletions()
                .withDefaultParametrizationsFirst()
                .getAlgorithm();

        hasco.setTimeout(TIMEOUT);


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
            if (s.getScore() > solution.getScore())
                solution = s;
        }

        writeToDatabase(new BenchmarkResult(
                DATASET_NAMES.get(i),
                Algos.HASCO,
                solution.getScore(), // equal to error rate here
                j
        ));
    }

    private void runMLPlan(ILabeledDataset ds, int i, int j) throws IOException, SplitFailedException, InterruptedException, DatasetDeserializationFailedException, URISyntaxException {

        List<ILabeledDataset<?>> split = SplitterUtil.getLabelStratifiedTrainTestSplit(ds, new Random(42), .7);  //TODO: parametrize seed
        MLPlan<IWekaClassifier> mlplan = new MLPlanWekaBuilder().withNumCpus(2).withTimeOut(TIMEOUT).withDataset(split.get(0)).build();

        long start;
        IWekaClassifier optimizedClassifier;
        try {
            start = System.currentTimeMillis();
            optimizedClassifier = mlplan.call();
            long trainTime = System.currentTimeMillis() - start;
            System.out.println("Finished build of the classifier. Training time was " + trainTime + "s");
            System.out.println("Chosen model is: " + mlplan.getSelectedClassifier());

            /* evaluate solution produced by mlplan */
            SupervisedLearnerExecutor executor = new SupervisedLearnerExecutor();
            ILearnerRunReport report = executor.execute(optimizedClassifier, split.get(1));
            double errorRate = EClassificationPerformanceMeasure.ERRORRATE.loss(report.getPredictionDiffList().getCastedView(Integer.class, ISingleLabelClassification.class));
            System.out.println("Error Rate of the solution produced by ML-Plan: " +
                    errorRate +
                    ". Internally believed error was " +
                    mlplan.getInternalValidationErrorOfSelectedClassifier()
            );

            writeToDatabase(new BenchmarkResult(
                    DATASET_NAMES.get(i),
                    Algos.MLPLAN,
                    errorRate,
                    j
            ));
        } catch (NoSuchElementException | AlgorithmException | InterruptedException | AlgorithmExecutionCanceledException | AlgorithmTimeoutedException | LearnerExecutionFailedException | SQLException e) {
            System.out.println("Building the classifier failed: " + e.getMessage());
        }

    }

    private List<File> datasetsAsFiles() {
        return DATASET_NAMES.stream().map(
                s -> {
                    try {
                        return new File(this.getClass().getClassLoader().getResource(s).toURI());
                    } catch (URISyntaxException e) {
                        e.printStackTrace();
                    }
                    return null;  // shouldn't happen
                }
        ).collect(Collectors.toList());
    }

    private List<ILabeledDataset<?>> labeledDatasets() {
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
}
