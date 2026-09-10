import { useQuery } from "@tanstack/react-query";
import { getSamples } from "../lib/api";

export function useSamples() {
  return useQuery({
    queryKey: ["samples"],
    queryFn: getSamples,
    staleTime: Infinity,
  });
}
