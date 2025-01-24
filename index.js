import Head from "next/head"
import { useState, useEffect, useRef } from "react"
import * as tf from "@tensorflow/tfjs"
import * as speechCommands from "@tensorflow-models/speech-commands"
import * as cocoSsd from "@tensorflow-models/coco-ssd"

export default function Home() {
  const [imageResult, setImageResult] = useState(null)
  const [audioResult, setAudioResult] = useState(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  useEffect(() => {
    const loadModels = async () => {
      await tf.ready()
      const audioModel = await speechCommands.create("BROWSER_FFT")
      await audioModel.ensureModelLoaded()

      const imageModel = await cocoSsd.load()

      startAudioRecognition(audioModel)
      startVideoRecognition(imageModel)
    }

    loadModels()
  }, [])

  const startAudioRecognition = async (model) => {
    const classLabels = model.wordLabels()
    model.listen(
      (result) => {
        const topResult = result.scores.indexOf(Math.max(...result.scores))
        setAudioResult(classLabels[topResult])
      },
      { probabilityThreshold: 0.7 },
    )
  }

  const startVideoRecognition = async (model) => {
    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas.getContext("2d")

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      video.srcObject = stream
      video.play()

      const detectFrame = async () => {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        const predictions = await model.detect(canvas)
        if (predictions.length > 0) {
          setImageResult(predictions[0].class)
        }
        requestAnimationFrame(detectFrame)
      }

      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        detectFrame()
      }
    }
  }

  return (
    <div className="min-h-screen bg-gray-100 py-6 flex flex-col justify-center sm:py-12">
      <Head>
        <title>Advanced AI Project</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="relative py-3 sm:max-w-xl sm:mx-auto">
        <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-light-blue-500 shadow-lg transform -skew-y-6 sm:skew-y-0 sm:-rotate-6 sm:rounded-3xl"></div>
        <div className="relative px-4 py-10 bg-white shadow-lg sm:rounded-3xl sm:p-20">
          <div className="max-w-md mx-auto">
            <div>
              <h1 className="text-2xl font-semibold">Advanced AI Project</h1>
            </div>
            <div className="divide-y divide-gray-200">
              <div className="py-8 text-base leading-6 space-y-4 text-gray-700 sm:text-lg sm:leading-7">
                <p>Audio Recognition Result: {audioResult || "Listening..."}</p>
                <p>Image Recognition Result: {imageResult || "Analyzing..."}</p>
                <div className="relative">
                  <video ref={videoRef} className="hidden" />
                  <canvas ref={canvasRef} className="w-full h-auto" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

