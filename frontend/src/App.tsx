import { BrowserRouter, Route, Routes } from "react-router-dom"
import { Layout } from "@/components/Layout"
import { ModelsProvider } from "@/models-context"
import { EvalPage } from "@/pages/EvalPage"
import { HowPage } from "@/pages/HowPage"
import { LearnPage } from "@/pages/LearnPage"
import { RecognizePage } from "@/pages/RecognizePage"
import { TrainPage } from "@/pages/TrainPage"

export default function App() {
  return (
    <ModelsProvider>
      <BrowserRouter basename={import.meta.env.BASE_URL.replace(/\/$/, "") || undefined}>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<RecognizePage />} />
            <Route path="learn" element={<LearnPage />} />
            <Route path="train" element={<TrainPage />} />
            <Route path="how" element={<HowPage />} />
            <Route path="evaluation" element={<EvalPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ModelsProvider>
  )
}
