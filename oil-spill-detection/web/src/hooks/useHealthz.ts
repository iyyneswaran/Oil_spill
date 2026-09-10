import { useQuery } from "@tanstack/react-query";
import { getHealth } from "../lib/api";

export function useHealthz() {
  return useQuery({
    queryKey: ["healthz"],
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: 1,
  });
}
