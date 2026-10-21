
using Microsoft.ML.Data;

namespace ForecastingAI;

public class StockData
{
    [LoadColumn(0)]
    public string Date { get; set; }
    
    [LoadColumn(1)]
    public float Open { get; set; }
    
    [LoadColumn(2)]
    public float High { get; set; }
    
    [LoadColumn(3)]
    public float Low { get; set; }
    
    [LoadColumn(4)]
    public float Close { get; set; }
}

public class StockPrediction
{
    [ColumnName("Score")]
    public float PredictedClose { get; set; }
}