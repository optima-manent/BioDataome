import { AtlasExplorer } from "./components/AtlasExplorer";
import { loadPublishedGraph } from "./lib/api-graph";

export default async function Home() {
  const graph = await loadPublishedGraph();
  return <AtlasExplorer graph={graph} />;
}
