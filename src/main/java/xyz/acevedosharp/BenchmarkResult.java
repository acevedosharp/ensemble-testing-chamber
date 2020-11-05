package xyz.acevedosharp;

import ai.libs.jaicore.components.model.ComponentInstance;

public class BenchmarkResult {
    public String datasetSource;
    public Algos algorithm;
    public Long trainTime;
    public Double errorRate;
    public String serializedComponentInstance;

    public BenchmarkResult(String datasetSource, Algos algorithm, Long trainTime, Double errorRate, String serializedComponentInstance) {
        this.datasetSource = datasetSource;
        this.algorithm = algorithm;
        this.trainTime = trainTime;
        this.errorRate = errorRate;
        this.serializedComponentInstance = serializedComponentInstance;
    }
}

