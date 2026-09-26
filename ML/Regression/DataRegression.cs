
namespace ML.Regression;

public static class DataRegression
{
    public static List<InsuranceData> GetInsuranceData()
    {
        return
        [
            new InsuranceData { Age = 10, Premium = 2000 },
            new InsuranceData { Age = 20, Premium = 2500 },
            new InsuranceData { Age = 30, Premium = 6000 },
            new InsuranceData { Age = 40, Premium = 9000 },
            new InsuranceData { Age = 50, Premium = 11000 },
            new InsuranceData { Age = 60, Premium = 12500 },
            new InsuranceData { Age = 70, Premium = 18000 },
            new InsuranceData { Age = 80, Premium = 19000 },
            new InsuranceData { Age = 90, Premium = 19500 },
            new InsuranceData { Age = 100, Premium = 21000 }
        ];
    }
}
