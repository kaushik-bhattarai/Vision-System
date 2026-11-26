"""People analytics on the final detections: dwell time per tracked person, crowds (nearby groups), clusters (many people)."""
from __future__ import annotations

import time

from .metrics import describe


class Analytics:
    def __init__(self, cams, a):
        self.cams = cams
        self.lost_s, self.alert_s = a.dwell_lost_s, a.dwell_alert_s
        self.crowd_n, self.crowd_dist, self.cluster_n = a.crowd_min_people, a.crowd_distance, a.cluster_min_people
        self.reset()

    def reset(self):
        n = len(self.cams)
        self.tracks = [{} for _ in range(n)]   # track id -> [first_seen, last_seen] (monotonic s)
        self.done = [[] for _ in range(n)]     # dwell of finished tracks
        self.st = [dict(frames=0, people_max=0, crowd_frames=0, crowd_max=0, crowd_episodes=0, crowd_now=False,
                        cluster_frames=0, cluster_episodes=0, cluster_now=False) for _ in range(n)]

    def _groups(self, people):
        """Connected components of 'near' people (union-find). Returns list of index lists."""
        n = len(people)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                _, xi, yi, hi = people[i]
                _, xj, yj, hj = people[j]
                d = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
                if d < self.crowd_dist * max((hi + hj) / 2.0, 1.0):
                    parent[find(i)] = find(j)
        comps = {}
        for i in range(n):
            comps.setdefault(find(i), []).append(i)
        return list(comps.values())

    def update(self, cam, people, now=None):
        """people = [(track_id or None, cx, cy, height)] of one camera frame (mux pixels).
        Returns per-person dwell (s or None), in_crowd flags, and the frame's crowd / cluster figures."""
        now = now if now is not None else time.monotonic()
        tr, s = self.tracks[cam], self.st[cam]
        for tid in [t for t, (_, last) in tr.items() if now - last > self.lost_s]:  # track ended
            f, last = tr.pop(tid)
            self.done[cam].append(last - f)
        dwell = []
        for tid, *_ in people:
            if tid is None:
                dwell.append(None)
                continue
            rec = tr.setdefault(tid, [now, now])
            rec[1] = now
            dwell.append(now - rec[0])

        crowds = [g for g in self._groups(people) if len(g) >= self.crowd_n]
        in_crowd = [False] * len(people)
        for g in crowds:
            for i in g:
                in_crowd[i] = True
        crowd_max = max((len(g) for g in crowds), default=0)
        cluster = len(people) >= self.cluster_n

        s["frames"] += 1
        s["people_max"] = max(s["people_max"], len(people))
        s["crowd_frames"] += bool(crowds)
        s["crowd_max"] = max(s["crowd_max"], crowd_max)
        s["crowd_episodes"] += bool(crowds) and not s["crowd_now"]
        s["crowd_now"] = bool(crowds)
        s["cluster_frames"] += cluster
        s["cluster_episodes"] += cluster and not s["cluster_now"]
        s["cluster_now"] = cluster
        return {"dwell": dwell, "in_crowd": in_crowd, "crowds": len(crowds), "crowd_max": crowd_max,
                "cluster": cluster, "people": len(people)}

    def summary(self):
        out = []
        for c, name in enumerate(self.cams):
            d = self.done[c] + [l - f for f, l in self.tracks[c].values()]
            s, n = self.st[c], max(self.st[c]["frames"], 1)
            out.append({"cam": name, "frames": s["frames"], "people_max": s["people_max"],
                        "crowd_frames_pct": 100.0 * s["crowd_frames"] / n, "crowd_max_size": s["crowd_max"],
                        "crowd_episodes": s["crowd_episodes"],
                        "cluster_frames_pct": 100.0 * s["cluster_frames"] / n,
                        "cluster_episodes": s["cluster_episodes"],
                        "dwell_s": describe(d), "tracks": len(d),
                        "loiterers": sum(x >= self.alert_s for x in d)})
        return out