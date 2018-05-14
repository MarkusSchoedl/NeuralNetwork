using System;
using System.Collections.Generic;
using System.Text;
using NUnit.Framework;

namespace NeuronalNetwork.UnitTesting
{
    [TestFixture]
    class MatrixMultTests
    {
        [Test]
        public void MatrixMultTest2By2With2By2()
        {
            Matrix m1 = new Matrix(2, 2);
            Matrix m2 = new Matrix(2, 2);

            m1.Value[0][0] = 13;
            m1.Value[0][1] = 32;
            m1.Value[1][0] = 65;
            m1.Value[1][1] = 11;

            m2.Value[0][0] = 35;
            m2.Value[0][1] = 12;
            m2.Value[1][0] = 29;
            m2.Value[1][1] = 73;

            Matrix result = Matrix.Multiply(m1, m2);

            Matrix target = new Matrix(2, 2);
            target.Value[0][0] = 1383;
            target.Value[0][1] = 2492;
            target.Value[1][0] = 2594;
            target.Value[1][1] = 1583;


            for (int i = 0; i < target.Rows; i++)
            {
                for (int j = 0; j < target.Columns; j++)
                {
                    Assert.That(Math.Abs(target.Value[i][j] - result.Value[i][j]) < 0.01);
                }
            }
        }

        [Test]
        public void MatrixMultTest2By2With2By1()
        {
            Matrix m1 = new Matrix(2, 2);
            Matrix m2 = new Matrix(2, 1);

            m1.Value[0][0] = 13;
            m1.Value[0][1] = 32;
            m1.Value[1][0] = 65;
            m1.Value[1][1] = 11;

            m2.Value[0][0] = 35;
            m2.Value[1][0] = 3;

            Matrix result = Matrix.Multiply(m1, m2);

            Assert.That(m1.Columns == 2 && m1.Rows == 2);
            Assert.That(m2.Columns == 1 && m2.Rows == 2);
            Assert.That(result.Columns == 1 && result.Rows == 2);

            Assert.That(result.Value[0][0] == 551);
            Assert.That(result.Value[1][0] == 2308);
        }

        [Test]
        public void MatrixMultTestNonStaticNormal()
        {
            Matrix m1 = new Matrix(3, 2);

            m1.Value[0][0] = 13;
            m1.Value[0][1] = 32;
            m1.Value[1][0] = -65;
            m1.Value[1][1] = 11;
            m1.Value[2][0] = 3;
            m1.Value[2][1] = 0;

            m1.Multiply(3);

            Assert.That(m1.Value[0][0] == 39);
            Assert.That(m1.Value[0][1] == 96);
            Assert.That(m1.Value[1][0] == -195);
            Assert.That(m1.Value[1][1] == 33);
            Assert.That(m1.Value[2][0] == 9);
            Assert.That(m1.Value[2][1] == 0);
        }

        [Test]
        public void MatrixMultTestNonStaticSmall()
        {
            Matrix m1 = new Matrix(3, 2);

            m1.Value[0][0] = 13;
            m1.Value[0][1] = 32;
            m1.Value[1][0] = -65;
            m1.Value[1][1] = 11;
            m1.Value[2][0] = 3;
            m1.Value[2][1] = 0;

            m1.Multiply(0.1);

            Assert.That(Math.Abs(m1.Value[0][0] - 1.3) < 0.01);
            Assert.That(Math.Abs(m1.Value[0][1] - 3.2) < 0.01);
            Assert.That(Math.Abs(m1.Value[1][0] - (-6.5)) < 0.01);
            Assert.That(Math.Abs(m1.Value[1][1] - 1.1) < 0.01);
            Assert.That(Math.Abs(m1.Value[2][0] - .3) < 0.01);
            Assert.That(Math.Abs(m1.Value[2][1]) < 0.01);
        }

        [Test]
        public void MatrixMultTestNonStaticNegative()
        {
            Matrix m1 = new Matrix(3, 2);

            m1.Value[0][0] = 13;
            m1.Value[0][1] = 32;
            m1.Value[1][0] = -65;
            m1.Value[1][1] = 11;
            m1.Value[2][0] = 3;
            m1.Value[2][1] = 0;

            m1.Multiply(-2);

            Assert.That(m1.Value[0][0] == -26);
            Assert.That(m1.Value[0][1] == -64);
            Assert.That(m1.Value[1][0] == 130);
            Assert.That(m1.Value[1][1] == -22);
            Assert.That(m1.Value[2][0] == -6);
            Assert.That(m1.Value[2][1] == 0);
        }

        [Test]
        public void MatrixMultTestElementwiseMult()
        {
            Matrix m1 = new Matrix(3, 1);
            Matrix m2 = new Matrix(3, 1);

            m1.Value[0][0] = 13;
            m1.Value[1][0] = -65;
            m1.Value[2][0] = 3;

            m2.Value[0][0] = 23;
            m2.Value[1][0] = -1;
            m2.Value[2][0] = 99;

            var res = Matrix.Multiply(m1, m2);

            Assert.That(Math.Abs(res.Value[0][0] - 13 * 23) < 0.001);
            Assert.That(Math.Abs(res.Value[1][0] - 65) < 0.001);
            Assert.That(Math.Abs(res.Value[2][0] - 99 * 3) < 0.001);
        }
    }
}
