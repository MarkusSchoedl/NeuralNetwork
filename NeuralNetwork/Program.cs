using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using static NeuronalNetwork.GetImages;

namespace NeuronalNetwork
{
    class Program
    {
        private const double DesiredEpsilon = 0.001;

        static void Main(string[] args)
        {
            string location = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location);
            var images = GetImagesFromFile(location, "train", 60000);

            Stopwatch watch = new Stopwatch();
            watch.Start();

            long iterations = 0;
            long errors = 0;
            double errorRate = 1;
            double epsilon = 1;

            Random rand = new Random();
            var nn = new NeuronalNetwork(784, 89, 10);
            double[] target = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

            // Train Network until desired epsilon is reached
            while (iterations <= 60000)
            {
                iterations++;
                int randInd = rand.Next(images.Count);

                int label = images[randInd].GetLabel();

                target[label] = 1;
                var res = nn.Train(images[randInd].GetData(), target);
                target[label] = 0;

                if (label != res)
                {
                    errors++;
                }

                if (iterations % 1000 == 0)
                {
                    errorRate = errors / 1000.0;
                    errors = 0;

                    if (label != res)
                    {
                        Console.ForegroundColor = ConsoleColor.DarkRed;
                    }

                    epsilon = nn.CalculateEpsilon(label);

                    Console.Write("Expected: " + label + " -  was: " + res + "   errorRate: " + errorRate.ToString("0.0000") + " Epsilon: " + epsilon.ToString("0.000000") + "\n");

                    Console.Write("\n");
                    Console.ForegroundColor = ConsoleColor.White;
                }
            }

            watch.Stop();
            var ts = watch.Elapsed;
            string elapsedTime = $"{ts.Hours} Hours, {ts.Minutes} Minutes, {ts.Seconds} Seconds, {ts.Milliseconds} ms";

            images.Clear();
            images = GetImagesFromFile(location, "t10k", 10000);

            errors = 0;
            errorRate = 1;

            int[,] confusionMatrix = new int[10, 10];

            for (int i = 0; i < images.Count; i++)
            {
                int res = nn.FeedForward(images[i].GetData());
                int label = images[i].GetLabel();

                if (res != label)
                {
                    errors++;
                }

                confusionMatrix[label, res]++;
            }

            string timePerPicture = (ts.TotalMilliseconds / iterations).ToString("N2");
            PrintResult(confusionMatrix, timePerPicture, elapsedTime);

            Console.ReadKey();
        }

        /// <summary>
        /// Prints all the relevant data into a beautiful matrix.
        /// </summary>
        /// <param name="confusionMatrix">The confusion matrix as result.</param>
        /// <param name="timePerPicture">The string formatted value of the average Time per picture</param>
        /// <param name="elapsedTime">The elapsed Time as string.</param>
        private static void PrintResult(int[,] confusionMatrix, string timePerPicture, string elapsedTime)
        {
            int rightGuesses = 0, wrongGuesses = 0;

            char[] yAxisLabel = "GUESSED".ToArray();

            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("\n\nElapsed Time: " + elapsedTime + " sec\n(" + timePerPicture + " ms per image)\n\n");

            Console.WriteLine("Confusion Matrix:");

            for (int i = 0; i < 10; i++)
            {
                // Print Axis Label
                if (i == 0)
                {
                    Console.Write("\n\n                      D E S I R E D   O U T P U T\n\n");
                    Console.Write("              ");
                    for (int j = 0; j < 10; j++)
                    {
                        Console.Write(" " + j + "   ");
                    }
                    Console.Write("\n");
                }

                // Print Y axis label
                if (i > 0 && i - 1 < yAxisLabel.Count())
                {
                    var oldCol = Console.ForegroundColor;
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Write(" " + yAxisLabel[i - 1] + "  ");
                    Console.ForegroundColor = oldCol;
                }
                else
                {
                    Console.Write("    ");
                }

                Console.ForegroundColor = ConsoleColor.White;
                Console.Write("Number " + i + ": ");
                for (int j = 0; j < 10; j++)
                {
                    if (i == j)
                    {
                        Console.ForegroundColor = ConsoleColor.Magenta;
                        rightGuesses += confusionMatrix[i, j];
                    }
                    else if (confusionMatrix[i, j] != 0)
                    {
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        wrongGuesses += confusionMatrix[i, j];
                    }
                    else
                    {
                        Console.ForegroundColor = ConsoleColor.White;
                    }

                    Console.Write(confusionMatrix[i, j].ToString("D4") + " ");
                }

                Console.Write("\n");
            }

            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("\n Wrong Guesses: " + wrongGuesses);
            Console.WriteLine(" Right Guesses: " + rightGuesses);
            Console.WriteLine(" Accuracy: " + ((rightGuesses / (float)(wrongGuesses + rightGuesses)) * 100.0).ToString("0.00") + "%");
        }
    }
}
