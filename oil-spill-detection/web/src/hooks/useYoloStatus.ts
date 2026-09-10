import { useQuery } from "@tanstack/react-query";
import { getYoloStatus } from "../lib/api";

export function useYoloStatus() {
  return useQuery({
    queryKey: ["yoloStatus"],
    queryFn: getYoloStatus,
    refetchInterval: 30_000,
    retry: 1,
  });
}
