package main.scala

import main.scala.obj.Model
import main.java.commons.cli.CommandLine
import main.scala.connector.Dictionary2File
import main.scala.obj.Parameter
import java.io.File
import main.scala.connector.Model2File
import main.scala.helper.Conversion
import main.scala.obj.Parameter
import scala.util.control.Breaks._

class SrcEstimator {
  // output model
  var trnModel: Model = null

  def init(params: Parameter): Boolean = {
    trnModel = new Model()

    if (!trnModel.initNewModel(params))
      false
    Dictionary2File.writeWordMap(params.directory + File.separator + "output" + File.separator + params.wordMapFileName, trnModel.data.localDict.word2id)

    true
  }

  def estimate(savestep: Int): Unit = {
    println("Sampling " + trnModel.niters + " iteration!")

    val lastIter = trnModel.liter
    val nextLastIter = trnModel.niters + lastIter
    for (iter <- (lastIter + 1) to nextLastIter) {
      println("Iteration " + iter + "...")

      // for all z_i
      for (m <- 0 until trnModel.M) {
        for (n <- 0 until trnModel.data.docs(m).length) {
          // z_i = z[m][n]
          // sample from p(z_i|z_-i, w)
          val topic = sampling(m, n)
          trnModel.z(m).update(n, topic)
        } // end for each word
      } // end for each document

      trnModel.liter = iter
      if (savestep > 0) {
        if (iter % savestep == 0 && iter < nextLastIter) {
          println("Saving the model at iteration " + iter + "...")
          computeTheta()
          computePhi()
          Model2File.saveModel("Src" + trnModel.modelName + "-" + Conversion.zeroPad(iter, 5), trnModel)
        }
      }
    } // end iterations	

    System.out.println("Gibbs sampling completed!\n")
    System.out.println("Saving the final model!\n")
    computeTheta()
    computePhi()
    //trnModel.liter -= 1
    Model2File.saveModel("Src" + trnModel.modelName + "-final", trnModel)
  }

  /**
   * Do sampling
   * @param m document number
   * @param n word number
   * @return topic id
   */
  def sampling(m: Int, n: Int): Int = {
    // remove z_i from the count variable
    var topic = trnModel.z(m)(n)
    val w = trnModel.data.docs(m).words(n)

    trnModel.nw(w)(topic) -= 1
    trnModel.nd(m)(topic) -= 1
    trnModel.nwsum(topic) -= 1
    trnModel.ndsum(m) -= 1

    val Vbeta = trnModel.V * trnModel.beta
    val Kalpha = trnModel.K * trnModel.alpha

    //do multinominal sampling via cumulative method
    for (k <- 0 until trnModel.K) {
      /*println("m: " + m + " n: " + n + " k: " + k)
      println(trnModel.nw(w)(k))
      println(trnModel.ks.deltaPow(k)(n))
      println(trnModel.nwsum(k) + trnModel.ks.deltaPowSum(k))
      println(trnModel.nd(m)(k) + trnModel.alpha)
      println(trnModel.ndsum(m) + Kalpha)*/

      trnModel.p(k) = (trnModel.nw(w)(k) + trnModel.ks.deltaPow(k)(n)) / (trnModel.nwsum(k) + trnModel.ks.deltaPowSum(k)) *
        (trnModel.nd(m)(k) + trnModel.alpha) / (trnModel.ndsum(m) + Kalpha)
    }

    // cumulate multinomial parameters
    for (k <- 1 until trnModel.K) {
      trnModel.p(k) += trnModel.p(k - 1)
    }

    // scaled sample because of unnormalized p[]
    val scale = Math.random() * trnModel.p(trnModel.K - 1)

    //sample topic w.r.t distribution p
    topic = 0
    if (trnModel.p(0) <= scale) {
      var low = 0
      var high = trnModel.K - 1
      breakable {
        while (low <= high) {
          if (low == high - 1) {
            topic = high
            break
          }

          val mid = (low + high) / 2
          if (trnModel.p(mid) > scale) high = mid
          else low = mid
        }
      }
    }
    /*for (topic <- 0 until trnModel.K) {
      if (trnModel.p(topic) > u) //sample topic w.r.t distribution p
        break
    }*/

    // add newly estimated z_i to count variables
    trnModel.nw(w)(topic) += 1;
    trnModel.nd(m)(topic) += 1;
    trnModel.nwsum(topic) += 1;
    trnModel.ndsum(m) += 1;

    topic
  }

  def computeTheta(): Unit = {
    for (m <- 0 until trnModel.M) {
      for (k <- 0 until trnModel.K) {
        trnModel.theta(m)(k) = (trnModel.nd(m)(k) + trnModel.alpha) / (trnModel.ndsum(m) + trnModel.K * trnModel.alpha)
      }
    }
  }

  def computePhi(): Unit = {
    for (k <- 0 until trnModel.K) {
      for (w <- 0 until trnModel.V) {
        trnModel.phi(k)(w) = (trnModel.nw(w)(k) + trnModel.beta) / (trnModel.nwsum(k) + trnModel.V * trnModel.beta)
      }
    }
  }
}