def set_matched_retrieval_deliberation_events(self, rec_min_free_before, rec_min_free_after,
                                              remove_around_recall=2000, match_tol=2000):
    """Sets matched recall/deliberation behavioral events to .events

    Parameters
    ----------
    rec_min_free_before: int, time in ms recall must be free prior to vocalization onset in order to be counted as a
                         included recall
    rec_min_free_after: int, time in ms recall must be free after vocalization onset in order to be counted as a
                        included recall
    remove_around_recall: int, by default, 2000, time in ms to remove as a valid deliberation period before and after
                          a vocalization
    match_tol: int, by default 2000, time in ms to tolerate as a possible recall deliberation match

    Creates
    -------
    Attribute events
    """
    evs = create_retrieval_and_matched_deliberation(subject=self.subject,
                                                    experiment=self.subject,
                                                    session=self.session,
                                                    rec_inclusion_before=rec_min_free_before,
                                                    rec_inclusion_after=rec_min_free_after,
                                                    remove_before_recall=remove_around_recall,
                                                    remove_after_recall=remove_around_recall,
                                                    match_tolerance=match_tol,
                                                    verbose=self.verbose)
    self.events = evs
    return
