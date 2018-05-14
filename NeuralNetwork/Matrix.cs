using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuronalNetwork
{
    class Matrix
    {
        private double[][] _matrix;
        private int _rows;
        private int _columns;

        public Matrix(int rows, int columns)
        {
            _rows = rows;
            _columns = columns;

            _matrix = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                _matrix[i] = new double[columns];
            }
        }

        /// <summary>
        /// Randomizes each Element of the matrix which sets it to a value between -1 and 1.
        /// </summary>
        public void Randomize()
        {
            Random randomizer = new Random();

            for (int row = 0; row < _rows; row++)
            {
                for (int col = 0; col < _columns; col++)
                {
                    _matrix[row][col] = randomizer.NextDouble() < 0.5
                        ? randomizer.NextDouble()
                        : randomizer.NextDouble() * -1;
                }
            }
        }

        public static Matrix FromArray(double[] array)
        {
            var result = new Matrix(array.Length, 1);

            for (int i = 0; i < array.Length; i++)
            {
                result.Value[i][0] = array[i];
            }

            return result;
        }

        public double[] ToArray()
        {
            double[] output = new double[_rows * _columns];
            int i = 0;

            for (int row = 0; row < _rows; row++)
            {
                for (int col = 0; col < _columns; col++, i++)
                {
                    output[i] = _matrix[row][col];
                }
            }

            return output;
        }

        public void Print()
        {
            for (int row = 0; row < _rows; row++)
            {
                for (int col = 0; col < _columns; col++)
                {
                    Console.Write(_matrix[row][col].ToString("+0.00;-0.00; 0.00") + "  ");
                }

                Console.WriteLine();
            }

        }

        #region GettersNSetters

        public double[][] Value
        {
            get => _matrix;
            set
            {
                _matrix = value;
                _rows = _matrix.Length;
                _columns = _matrix.First().Length;
            }
        }

        public int Rows => _rows;
        public int Columns => _columns;

        #endregion

        #region Math Functions
        /// <summary>
        /// Multiplies each element of the matrix with the given number.
        /// </summary>
        public void Multiply(double num)
        {
            for (int row = 0; row < _rows; row++)
            {
                for (int col = 0; col < _columns; col++)
                {
                    _matrix[row][col] *= num;
                }
            }
        }

        /// <summary>
        /// Calculates the hadamard Product.
        /// </summary>
        public void Multiply(Matrix m)
        {
            if (_rows != m.Rows || _columns != m.Columns)
            {
                throw new InvalidOperationException("The Rows and Columns didnt match.");
            }

            //multiply each element
            for (int row = 0; row < m.Rows; row++)
            {
                for (int col = 0; col < m.Columns; col++)
                {
                    //Calculate product
                    _matrix[row][col] *= m.Value[row][col];
                }
            }
        }

        /// <summary>
        /// Multiplies this matrix with another given matrix. 
        /// </summary>
        /// <exception cref="InvalidOperationException">This exception is thrown, in case the given matrix rows and columns don't match the rows and columns of this object.</exception>
        public static Matrix Multiply(Matrix m, Matrix m2)
        {
            if (m2.Rows != m.Columns)
            {
                throw new InvalidOperationException("The Rows and Columns didnt match.");
            }

            var result = new Matrix(m.Rows, m2.Columns);

            for (int row = 0; row < result.Rows; row++)
            {
                for (int col = 0; col < result.Columns; col++)
                {
                    //Calculate product
                    for (int i = 0; i < m.Columns; i++)
                    {
                        result.Value[row][col] += m.Value[row][i] * m2.Value[i][col];
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Multiplies this matrix with another given matrix. 
        /// </summary>
        /// <exception cref="InvalidOperationException">This exception is thrown, in case the given matrix rows and columns don't match the rows and columns of this object.</exception>
        public static Matrix Multiply(Matrix m, double[] array)
        {
            if (array.Length != m.Columns)
            {
                throw new InvalidOperationException("The Rows and Columns didnt match.");
            }

            var result = new Matrix(m.Rows, 1);

            for (int row = 0; row < result.Rows; row++)
            {
                for (int col = 0; col < result.Columns; col++)
                {
                    //Calculate product
                    result.Value[row][col] = m.Value[row][0] * array[col];
                }
            }

            return result;
        }

        /// <summary>
        /// Adds a number to each element of the matrix.
        /// </summary>
        public void Add(double num)
        {
            for (int row = 0; row < _rows; row++)
            {
                for (int col = 0; col < _columns; col++)
                {
                    _matrix[row][col] += num;
                }
            }
        }

        /// <summary>
        /// Adds another matrix to this matrix.
        /// </summary>
        /// <exception cref="InvalidOperationException">This exception is thrown, if the columns and rows of the given matrix are not the same.</exception>
        public void Add(Matrix m)
        {
            if (Rows != m.Rows && Columns != m.Columns)
            {
                throw new InvalidOperationException("The Rows and Columns didnt match.");
            }

            for (int row = 0; row < m.Rows; row++)
            {
                for (int col = 0; col < m.Columns; col++)
                {
                    Value[row][col] += m.Value[row][col];
                }
            }
        }

        /// <summary>
        /// Uses the given Function to apply it to each element on the Matrix.
        /// </summary>
        public void Map(Func<double, float> method)
        {
            for (int row = 0; row < Rows; row++)
            {
                for (int col = 0; col < Columns; col++)
                {
                    Value[row][col] = method(Value[row][col]);
                }
            }
        }

        /// <summary>
        /// Uses the given Function to apply it to each element on the Matrix.
        /// </summary>
        public static Matrix Map(Matrix m, Func<double, float> method)
        {
            var output = new Matrix(m.Rows, m.Columns);

            for (int row = 0; row < m.Rows; row++)
            {
                for (int col = 0; col < m.Columns; col++)
                {
                    output.Value[row][col] = method(m.Value[row][col]);
                }
            }

            return output;
        }

        /// <summary>
        /// Switches the matrixs Rows and Columns and returns a new Matrix.
        /// </summary>
        public static Matrix Transpose(Matrix m)
        {
            var result = new Matrix(m.Columns, m.Rows);

            for (int row = 0; row < m.Rows; row++)
            {
                for (int col = 0; col < m.Columns; col++)
                {
                    result.Value[col][row] = m.Value[row][col];
                }
            }

            return result;
        }

        public static Matrix Substract(Matrix m1, Matrix m2)
        {
            if (m2.Rows != m1.Rows && m2.Columns != m1.Columns)
            {
                throw new InvalidOperationException("The Rows and Columns didnt match.");
            }

            var result = new Matrix(m1.Rows, m1.Columns);

            for (int row = 0; row < m1.Rows; row++)
            {
                for (int col = 0; col < m1.Columns; col++)
                {
                    result.Value[row][col] = Convert.ToDouble(m1.Value[row][col]) - Convert.ToDouble(m2.Value[row][col]);
                }
            }

            return result;
        }
        #endregion
    }
}