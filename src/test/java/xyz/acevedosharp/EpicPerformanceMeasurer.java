package xyz.acevedosharp;

import ai.libs.hasco.core.events.HASCOSolutionEvent;
import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.components.model.RefinementConfiguredSoftwareConfigurationProblem;

import java.io.*;
import java.net.URISyntaxException;
import java.sql.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.TimeUnit;

import ai.libs.hasco.builder.HASCOBuilder;
import ai.libs.hasco.builder.forwarddecomposition.HASCOViaFD;
import ai.libs.hasco.core.HASCOSolutionCandidate;

import ai.libs.jaicore.ml.core.dataset.serialization.ArffDatasetAdapter;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import ai.libs.mlplan.weka.MLPlanWekaBuilder;

import org.api4.java.ai.ml.core.dataset.serialization.DatasetDeserializationFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.algorithm.Timeout;

import org.api4.java.algorithm.events.IAlgorithmEvent;
import org.api4.java.algorithm.exceptions.AlgorithmException;
import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.api4.java.common.attributedobjects.IObjectEvaluator;
import org.junit.Test;

import java.io.File;
import java.util.stream.Collectors;

@SuppressWarnings({"unchecked", "ConstantConditions", "rawtypes"})
public class EpicPerformanceMeasurer {

    private static final List<String> DATASET_NAMES = Arrays.asList("datasets/iris.arff");
    private static final Timeout TIMEOUT = new Timeout(600, TimeUnit.SECONDS);
    private static final int REPETITIONS = 1;
    private static final int CORES = 3;
    @Test
    public void testMaxMemory() {
    }

    @Test
    public void runBoth() throws Exception {
        System.out.println("Max memory" + Runtime.getRuntime().maxMemory());
        epicHascoRunner();
        epicMLPlanRunner();
    }

    @Test
    public void epicHascoRunner() throws Exception {
        List<ILabeledDataset<?>> datasets = loadLabeledDatasets();
        for (int datasetIndex = 0; datasetIndex < DATASET_NAMES.size(); datasetIndex++) {
            for (int repetition = 1; repetition <= REPETITIONS; repetition++) {
                // split dataset
                List<ILabeledDataset> split = SplitterUtil.getLabelStratifiedTrainTestSplit(datasets.get(datasetIndex), repetition, .7);

                ILabeledDataset trainSet = split.get(0);
                ILabeledDataset testSet = split.get(1);

                // run the algorithm with .7 from above
                IWekaClassifier result = runHasco(trainSet, datasetIndex, repetition);

                // evaluate its result
                Double performance = EpicEnsembleEvaluator.measureAlgorithmPerformance(result, trainSet, testSet);

                // write result to database
                writeToDatabase(new BenchmarkResult(
                        DATASET_NAMES.get(datasetIndex),
                        Algos.HASCO,
                        performance,
                        repetition
                ));
            }
        }
    }

    @Test
    public void epicMLPlanRunner() throws Exception {
        List<ILabeledDataset<?>> datasets = loadLabeledDatasets();
        for (int datasetIndex = 0; datasetIndex < DATASET_NAMES.size(); datasetIndex++) {
            for (int repetition = 1; repetition <= REPETITIONS; repetition++) {
                // split dataset
                List<ILabeledDataset> split = SplitterUtil.getLabelStratifiedTrainTestSplit(datasets.get(datasetIndex), repetition, .7);

                // run the algorithm with .7 from above
                IWekaClassifier optimizedClassifier = runMLPlan(split.get(0));

                // evaluate its result with .3
                Double performance = EpicEnsembleEvaluator.measureAlgorithmPerformance(optimizedClassifier, split.get(0), split.get(1));

                // SUGGESTION: store results for every classifier and compare that against the ensemble

                // write result to database
                writeToDatabase(new BenchmarkResult(
                        DATASET_NAMES.get(datasetIndex),
                        Algos.MLPLAN,
                        performance,
                        repetition
                ));
            }
        }
    }

    private IWekaClassifier runHasco(ILabeledDataset dataset, int datasetIndex, int repetition) throws Exception {
        System.out.println("Execution of HASCO #"+repetition+" on dataset: "+ DATASET_NAMES.get(datasetIndex) + " began at "+ LocalDateTime.now().format(DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm:ss"))+".");

        String reqInterface = "EpicEnsemble";
        File componentFile = new File(this.getClass().getClassLoader().getResource("search-space/ensemble-configuration.json").toURI());

        IObjectEvaluator<ComponentInstance, Double> evaluator = new EpicEnsembleEvaluator(dataset, repetition);

        RefinementConfiguredSoftwareConfigurationProblem<Double> problem =
                new RefinementConfiguredSoftwareConfigurationProblem(componentFile, reqInterface, evaluator);

        HASCOViaFD<Double> hasco = HASCOBuilder.get()
                .withProblem(problem)
                .withBestFirst()
                .withCPUs(CORES)
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

        return EpicEnsembleFactory.getEnsemble(solution.getComponentInstance());
    }

    private IWekaClassifier runMLPlan(ILabeledDataset dataset) throws IOException, InterruptedException, AlgorithmExecutionCanceledException, AlgorithmTimeoutedException, AlgorithmException {
        // train with the entire dataset (0.7 from runner), the split and benchmarking is handled in the runner.
        return new MLPlanWekaBuilder().withNumCpus(CORES).withTimeOut(TIMEOUT).withDataset(dataset).build().call();
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
