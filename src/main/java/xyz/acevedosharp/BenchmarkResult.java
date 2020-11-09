package xyz.acevedosharp;

import ai.libs.jaicore.components.model.ComponentInstance;

public class BenchmarkResult {
    public String datasetSource;
    public Algos algorithm;
    public Double errorRate;
    public Integer repetition;

    public BenchmarkResult(String datasetSource, Algos algorithm, Double errorRate, Integer repetition) {
        this.datasetSource = datasetSource;
        this.algorithm = algorithm;
        this.errorRate = errorRate;
        this.repetition = repetition;
    }
}

