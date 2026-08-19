---
'@xxscreeps/pathfinder': patch
---

Keep multi-goal search goals alive for the whole search. `heuristic_t::make_from_runtime` returned a `std::span` into a vector local to that call, so every search with two or more goals evaluated its heuristic against freed memory. Reused memory usually reads as "the origin already satisfies a goal", which returns `ops=0`, `cost=0`, an empty path, and `incomplete=false` without ever invoking the room callback. The goal storage is now owned by the caller that runs the search.
