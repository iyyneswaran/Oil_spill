import { useQuery } from "@tanstack/react-query";
import { getModels } from "../lib/api";

export function useModels() {
  return useQuery({
    queryKey: ["models"],
    queryFn: getModels,
    staleTime: Infinity,
  });
}
