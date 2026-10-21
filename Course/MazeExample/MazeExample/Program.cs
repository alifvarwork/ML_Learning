using TorchSharp;

int[,] maze1 = {
    //0   1   2   3   4   5   6   7   8   9   10  11
    { 0 , 0 , 0 , 0 , 0 , 2 , 0 , 0 , 0 , 0 , 0 , 0 }, //row 0
    { 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 0 }, //row 1
    { 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 }, //row 2
    { 0 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 3
    { 0 , 0 , 0 , 0 , 1 , 1 , 0 , 1 , 0 , 1 , 1 , 0 }, //row 4
    { 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 }, //row 5
    { 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 1 , 0 }, //row 6
    { 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 }, //row 7
    { 0 , 1 , 0 , 1 , 1 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 8
    { 0 , 1 , 0 , 1 , 0 , 0 , 0 , 1 , 0 , 1 , 1 , 0 }, //row 9
    { 0 , 1 , 1 , 1 , 0 , 1 , 1 , 1 , 0 , 1 , 1 , 0 }, //row 10
    { 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0, 0 }  //row 11 (start position is (11, 5))
};

const string UP = "up";
const string DOWN = "down";
const string LEFT = "left";
const string RIGHT = "right";
string[] actions = { UP, DOWN, LEFT, RIGHT };

int[,] rewards;

const int WALL_REWARD_VALUE = -500;
const int FLOOR_REWARD_VALUE = -10;
const int GOAL_REWARD_VALUE = 500;

void SetupRewards(int[,] maze, int wallValue, int floorValue, int goalValue)
{
    int mazeRows = maze1.GetLength(0);
    int mazeCols = maze1.GetLength(1);
    rewards = new int[mazeRows, mazeCols];

    for (int i = 0; i < mazeRows; i++)
    {
        for (int j = 0; j < mazeCols; j++)
        {
            if (maze[i, j] == 0)
                rewards[i, j] = wallValue;
            else if (maze[i, j] == 1)
                rewards[i, j] = floorValue;
            else if (maze[i, j] == 2)
                rewards[i, j] = goalValue;
        }
    }
}

torch.Tensor qValues;

void SetupQValues(int[,] maze)
{
    int mazeRows = maze.GetLength(0);
    int mazeCols = maze.GetLength(1);

    qValues = torch.zeros(mazeRows, mazeCols, 4);
}

bool HasHitWallOrEndOfMaze(int curretRow, int currentColumn, int floorValue)
{
    return rewards[curretRow, currentColumn] != floorValue;
}

long DetermineNextAction(int currentRow, int currentColumn, float epsilon)
{
    Random random = new();
    double randomValue = random.NextDouble();
    long nextAction = randomValue < epsilon ? torch.argmax(qValues[currentRow, currentColumn]).item<long>() : random.Next(4);
    return nextAction;
}

(int, int) MoveOneSpace(int[,] maze, int currentRow, int currentColumn, long action)
{
    var mazeRows = maze.GetLength(0);
    var mazeCols = maze.GetLength(1);

    int newRow = currentRow;
    int newColumn = currentColumn;
    if (actions[action] == UP && currentRow > 0)
        newRow--;
    else if (actions[action] == DOWN && currentRow < mazeRows - 1) // DOWN
        newRow++;
    else if (actions[action] == LEFT && currentColumn > 0) // LEFT
        newColumn--;
    else if (actions[action] == RIGHT && currentColumn < mazeCols - 1) // RIGHT
        newColumn++;
    
    return (newRow, newColumn);
}

void TrainTheModel(int[,] maze, int floorValue, float epsilon, float discountFactor, float learningRate, int episodes)
{
    var mazeRows = maze.GetLength(0);
    var mazeCols = maze.GetLength(1);
    for (int episode = 0; episode < episodes; episode++)
    {
        Console.WriteLine("----- Starting Episode " + episode + " -----");

        int currentRow = 11;
        int currentColumn = 5;

        while (!HasHitWallOrEndOfMaze(currentRow, currentColumn, floorValue))
        {
            long action = DetermineNextAction(currentRow, currentColumn, epsilon);
            var previousRow = currentRow;
            var previousColumn = currentColumn;
            (int newRow, int newColumn) = MoveOneSpace(maze, currentRow, currentColumn, action);
            currentRow = newRow;
            currentColumn = newColumn;
            
            float reward = rewards[currentRow, currentColumn];
            float oldQValue = qValues[previousRow, previousColumn, action].item<float>();
            float temporalDifference = reward + torch.max(qValues[currentRow, currentColumn]).item<float>() - oldQValue;
            float newQValue = oldQValue + (learningRate * temporalDifference);

            qValues[previousRow, previousColumn, action] = newQValue;

            //if (HasHitWallOrEndOfMaze(newRow, newColumn, floorValue))
            //    break;
        }

        Console.WriteLine("----- Ending Episode " + episode + " -----");
    }
    Console.WriteLine("Training complete.");
}

List<int[]> NavigateMaze(int[,] maze, int startRow, int startColumn, int floorValue, int wallValue)
{
    List<int[]> path = [] ;
    int currentRow = startRow;
    int currentColumn = startColumn;
    if(HasHitWallOrEndOfMaze(currentRow, currentColumn, floorValue))
    {
        Console.WriteLine("Starting position is invalid.");
        return path;
    }

    path.Add([currentRow, currentColumn]);
    while (!HasHitWallOrEndOfMaze(currentRow, currentColumn, floorValue))
    {
        int nextAction = (int)DetermineNextAction(currentRow, currentColumn, 1.0f);
        (int newRow, int newColumn) = MoveOneSpace(maze, currentRow, currentColumn, nextAction);
        currentRow = newRow;
        currentColumn = newColumn;
        if(rewards[currentRow, currentColumn] == wallValue)
        {
            Console.WriteLine("Hit a wall at position: (" + currentRow + ", " + currentColumn + ")");
            break;
        }
        path.Add([newRow, newColumn]);
    }
    int moveCount = 1;
    for(int i = 0;  i < path.Count; i++)
    {
        Console.Write("Move " + moveCount + ": (");
        foreach (var element in path[i])
        {
            Console.Write(element + ", ");
        }
        Console.WriteLine(")");
        moveCount++;
    }
    return path;
}

const float EPSILON = 0.95f;
const float DISCOUNT_FLOOR = 0.8f;
const float LEARNING_RATE = 0.9f;
const int EPISODES = 1500;
const int START_ROW = 11;
const int START_COLUMN = 5;

SetupRewards(maze1, WALL_REWARD_VALUE, FLOOR_REWARD_VALUE, GOAL_REWARD_VALUE);
SetupQValues(maze1);
TrainTheModel(maze1, FLOOR_REWARD_VALUE, EPSILON, DISCOUNT_FLOOR, LEARNING_RATE, EPISODES);
NavigateMaze(maze1, START_ROW, START_COLUMN, FLOOR_REWARD_VALUE, WALL_REWARD_VALUE);