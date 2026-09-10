import { useMutation } from "@tanstack/react-query";
import { predict } from "../lib/api";

interface PredictVars {
  file: Blob;
  model: string;
  filename?: string;
}

export function usePredict() {
  return useMutation({
    mutationFn: ({ file, model, filename }: PredictVars) =>
      predict(file, model, filename),
  });
}
