package xyz.acevedosharp;

public class BenchmarkResult {
    public Long trainTime;
    public Double errorRate;

    public BenchmarkResult(Long trainTime, Double errorRate) {
        this.trainTime = trainTime;
        this.errorRate = errorRate;
    }
}
