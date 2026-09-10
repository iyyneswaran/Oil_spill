import { useMutation, useQuery } from "@tanstack/react-query";
import { createSceneJob, getJob } from "../lib/api";
import type { SceneJobRequest, Job } from "../lib/types";
import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "oilspill_active_jobs";

function loadPersistedJobs(): string[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function persistJobs(ids: string[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(ids));
  } catch {
    /* localStorage may be unavailable */
  }
}

export function useSubmitSceneJob() {
  return useMutation({
    mutationFn: (body: SceneJobRequest) => createSceneJob(body),
  });
}

export function useJobStatus(jobId: string | null) {
  return useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const data = query.state.data as Job | undefined;
      if (!data) return 2000;
      if (data.status === "done" || data.status === "error") return false;
      // Back off when tab is hidden
      return document.hidden ? 10_000 : 2000;
    },
    retry: (failureCount, error) => {
      // Don't retry 404s — server may have restarted
      if (error instanceof Error && error.message.startsWith("404")) return false;
      return failureCount < 2;
    },
  });
}

/** Manage a list of active job IDs with localStorage persistence. */
export function useJobList() {
  const [jobIds, setJobIds] = useState<string[]>(() => loadPersistedJobs());

  useEffect(() => {
    persistJobs(jobIds);
  }, [jobIds]);

  const addJob = useCallback((id: string) => {
    setJobIds((prev) => [id, ...prev]);
  }, []);

  const removeJob = useCallback((id: string) => {
    setJobIds((prev) => prev.filter((j) => j !== id));
  }, []);

  return { jobIds, addJob, removeJob };
}
