using Microsoft.ML.Data;


namespace ML;

public class InsuranceData
{
    [LoadColumn(0)]
    public float Age { get; set; }
    [LoadColumn(1)]
    public float Premium { get; set; }
}

public class InsurancePrediction
{
    [ColumnName("Score")]
    public float PredictedPremium { get; set; }
}
