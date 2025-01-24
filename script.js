let audioModel, imageModel

async function loadModels() {
  try {
    await tf.ready()
    audioModel = await speechCommands.create("BROWSER_FFT")
    await audioModel.ensureModelLoaded()

    imageModel = await cocoSsd.load()

    startAudioRecognition()
    startVideoRecognition()
  } catch (error) {
    console.error("Error loading models:", error)
  }
}

async function startAudioRecognition() {
  const classLabels = audioModel.wordLabels()
  audioModel.listen(
    (result) => {
      const scores = result.scores
      const topResult = scores.indexOf(Math.max(...scores))
      document.getElementById("audioResult").textContent = classLabels[topResult]
    },
    { probabilityThreshold: 0.7 },
  )
}

async function startVideoRecognition() {
  const video = document.getElementById("video")
  const canvas = document.getElementById("canvas")
  const ctx = canvas.getContext("2d")

  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true })
    video.srcObject = stream
    video.play()

    async function detectFrame() {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      const predictions = await imageModel.detect(canvas)
      if (predictions.length > 0) {
        document.getElementById("imageResult").textContent = predictions[0].class
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

loadModels()

